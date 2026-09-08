"""Guided case selection, query definition, validation, and analysis."""

from __future__ import annotations

import streamlit as st

from app.runtime import catalog, runner
from app.views import ate, interference, regime
from cwfm.application import (
    ATEQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    InterferenceQuery,
    RegimeQuery,
    SupportState,
    Task,
)


TASK_METHOD = {
    "Static treatment effect": (Task.STATIC_ATE, "static_ate"),
    "Observed regime": (Task.OBSERVED_REGIME, "causal_model_determination"),
    "Network interference": (Task.NETWORK_INTERFERENCE, "causal_interference_detection"),
}


def _case_selector(method: str):
    source = catalog()
    scenarios = source.scenarios(method)
    scenario = st.selectbox("Scenario", scenarios)
    seeds = source.seeds(method, scenario)
    seed = st.selectbox("Seed", seeds, format_func=lambda value: f"{value:04d}")
    return source.resolve(method, scenario, seed)


def _show_result(result, command: str, research_mode: bool) -> None:
    if result.task == Task.STATIC_ATE:
        ate.render(result, command, research_mode)
    elif result.task == Task.OBSERVED_REGIME:
        regime.render(result, command, research_mode)
    else:
        interference.render(result, command, research_mode)


def render() -> None:
    st.title("Analyze with CWFM")
    task_label = st.selectbox("Task", list(TASK_METHOD))
    task, method = TASK_METHOD[task_label]
    case = _case_selector(method)
    metadata = case.safe_metadata()
    preview = case.preview()
    with st.expander("Observed data preview", expanded=True):
        st.json(preview)
    st.caption(f"Backend: {case.backend}")

    features = tuple(metadata.get("feature_names", []))
    outcomes = tuple(metadata.get("outcome_names", ["Y"]))
    with st.form("analysis_form"):
        if task == Task.STATIC_ATE:
            outcome = st.selectbox("Outcome", outcomes)
            covariates = tuple(st.multiselect("Covariates", features, default=list(features)))
            assignment_default = metadata.get("assignment_design") == "randomized"
            assignment = st.selectbox(
                "Assignment design",
                list(AssignmentDesign),
                index=0 if assignment_default else 1,
                format_func=lambda value: value.value,
            )
            no_confounding = st.checkbox("Assert no unmeasured confounding", value=assignment_default)
            consistency = st.checkbox("Assert consistency", value=True)
            query = ATEQuery("A", outcome, covariates)
            assumptions = AssumptionLedger(
                assignment_design=assignment,
                no_unmeasured_confounding=AssertionState.ASSERTED if no_confounding else AssertionState.UNKNOWN,
                consistency=AssertionState.ASSERTED if consistency else AssertionState.UNKNOWN,
                treatment_support=SupportState.UNKNOWN,
            )
            command = f"python examples/01_static_ate.py --preset {case.scenario} --seed {case.seed}"
        elif task == Task.OBSERVED_REGIME:
            outcome = st.selectbox("Outcome", outcomes)
            default_ordinary = [features[index] for index in metadata.get("ordinary_predictors", [0, 1, 2, 3])]
            ordinary = tuple(st.multiselect("Ordinary predictors", features, default=default_ordinary))
            available = [name for name in features if name not in ordinary]
            policy = st.selectbox("Candidate-selection policy", ("screen-then-model", "explicit", "chunked"))
            candidates = tuple(st.multiselect("Candidate regime variables", available, default=available))
            budget = 12 - len(ordinary) - 1
            st.caption(f"Variable budget: {len(ordinary)} predictors + up to {budget} candidates + 1 outcome = 12")
            query = RegimeQuery(outcome, ordinary, candidates, candidate_policy=policy)
            assumptions = AssumptionLedger()
            command = f"python examples/02_observed_regimes.py --scenario {case.scenario} --seed {case.seed} --outcome {outcome} --candidate-policy {policy}"
        else:
            outcome = st.selectbox("Outcome", outcomes)
            covariates = tuple(st.multiselect("Covariates", features, default=list(features)))
            mapping = st.selectbox("Exposure mapping", ("weighted", "unweighted", "distance_two"))
            g0, g1 = st.slider("Exposure contrast", 0.0, 1.0, (0.25, 0.75), 0.05)
            assignment = st.selectbox("Assignment design", list(AssignmentDesign), format_func=lambda value: value.value)
            consistency = st.checkbox("Assert consistency", value=True)
            predeclared = st.checkbox("Mapping selected before outcome review", value=True)
            restricted = st.checkbox("Interference restricted to observed W", value=True)
            no_confounding = st.checkbox("Assert no unmeasured confounding", value=False)
            query = InterferenceQuery("A", "G", outcome, covariates, g0, g1, mapping)
            assumptions = AssumptionLedger(
                assignment_design=assignment,
                no_unmeasured_confounding=AssertionState.ASSERTED if no_confounding else AssertionState.UNKNOWN,
                consistency=AssertionState.ASSERTED if consistency else AssertionState.UNKNOWN,
                treatment_support=SupportState.UNKNOWN,
                exposure_mapping_predeclared=predeclared,
                interference_restricted_to_observed_graph=restricted,
            )
            command = f"python examples/03_network_interference.py --scenario {case.scenario} --seed {case.seed} --outcome {outcome} --mapping {mapping} --g0 {g0} --g1 {g1}"
        st.subheader("Assumption review")
        st.json({key: getattr(value, "value", value) for key, value in assumptions.__dict__.items()})
        submitted = st.form_submit_button("Run analysis", type="primary")

    if submitted:
        with st.spinner("Validating and analyzing…"):
            result = runner().analyze(case, query, assumptions)
        st.session_state["last_result"] = result
        st.session_state["last_command"] = command
        comparisons = st.session_state.setdefault("comparison_results", [])
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        research_mode = st.toggle("Research mode", value=False)
        _show_result(result, st.session_state.get("last_command", ""), research_mode)
        if st.button("Add current analysis to comparison"):
            existing = st.session_state.setdefault("comparison_results", [])
            if all(item.analysis_id != result.analysis_id for item in existing):
                existing.append(result)
                st.success("Added to comparison")
