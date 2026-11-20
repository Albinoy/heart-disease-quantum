"""
CardiaQ - Phase 1 Streamlit section: "Why this prediction?"

Use it from your existing app.py:

    from explain_tab import render_explanation

    # after you have the trained Random Forest (rf_model),
    # the training features (X_train) and the patient's input row (patient_df):
    render_explanation(rf_model, X_train, patient_df)

patient_df must be a ONE-row DataFrame with exactly the same columns
(and the same encoding / scaling) as X_train.
"""

import streamlit as st

from explain import (
    REGION_NAMES,
    build_explainer,
    caveats,
    explain_in_words,
    explain_patient,
    global_importance,
    plot_global,
    plot_patient,
    region_scores,
)


@st.cache_resource
def _cached_explainer(_model):
    # Leading underscore tells Streamlit not to try hashing the model.
    return build_explainer(_model)


@st.cache_data
def _cached_global(_model, _explainer, X_train, model_key: str):
    return global_importance(_model, _explainer, X_train)


def render_explanation(model, X_train, patient_df, model_key: str = "rf_v1"):
    """X_train is optional: pass None to skip the 'across all patients' chart."""
    explainer = _cached_explainer(model)
    result = explain_patient(model, explainer, patient_df)

    st.subheader("Why this prediction?")
    st.metric("Predicted heart disease risk", f"{result['probability']:.0%}")

    st.pyplot(plot_patient(result))
    st.caption("Red bars pushed the risk up for this patient, blue bars pushed it down.")

    st.markdown("**Main factors in plain English**")
    for line in explain_in_words(result):
        st.markdown(f"- {line}")
    for note in caveats(result):
        st.warning(note)

    # Preview of Phase 3: these are the regions the 3D heart will blink.
    regions = region_scores(result["contributions"], top_n=3)
    if len(regions):
        st.markdown("**Heart regions linked to this result** (these will blink in the 3D view)")
        for _, r in regions.iterrows():
            st.markdown(f"- **{r['region_name']}**: {r['features']}")

    with st.expander("Full contribution table"):
        table = result["contributions"].copy()
        table["shap"] = (table["shap"] * 100).round(2)
        table = table.rename(columns={"shap": "effect (pts)"})
        st.dataframe(table[["label", "value", "effect (pts)", "direction", "region"]], hide_index=True)

    if X_train is not None:
        with st.expander("What matters most across all patients"):
            st.pyplot(plot_global(_cached_global(model, explainer, X_train, model_key)))

    st.info(
        "CardiaQ is an educational tool, not a medical diagnosis. "
        "Explanations show what drove the model's output, not a confirmed condition."
    )
    return result
