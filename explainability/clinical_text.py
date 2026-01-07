def clinical_explanation(risk, contrib):
    if risk < 0.3:
        return (
            "The model detected minimal neurological risk indicators. "
            "Speech and handwriting patterns appear within normal variation."
        )

    if risk < 0.6:
        return (
            "The model identified mild deviations in speech and handwriting "
            "biomarkers that may warrant longitudinal monitoring."
        )

    return (
        "The model detected notable deviations in neurological biomarkers, "
        "primarily influenced by "
        f"{'speech' if contrib['speech'] > contrib['handwriting'] else 'handwriting'} patterns. "
        "This result is not diagnostic and should be interpreted cautiously."
    )
