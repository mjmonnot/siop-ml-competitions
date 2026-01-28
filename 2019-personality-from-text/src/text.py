def concat_text(df, text_cols):
    """
    Concatenate all open-ended responses into a single document.
    """
    return (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
