def preprocess_text(text, consider="lower"):
    """
    Input:
            - text: input to preprocess 
            - consider: Can take ["lower", "upper", "mean"]. This is used to preprocess range values given such as 60-65. 
                        if lower is taken then it will be converted to 60, if upper then to 65, if mean then 62.5
    """
    if isinstance(text, str):
        if '~' in text:
            text = text.replace("~", "")
        if "," in text:
            text = text.replace(",", ".")
        if not text.replace(" ", ""):
            return 0.0
        
        if '-' in text:
            lower, upper = text.split('-')
            if consider == "lower":
                return float(lower)
            elif consider == "upper":
                return float(upper)
            elif consider == "mean":
                mean = (float(lower) + float(upper)) / 2
                return mean
        else:
            return float(text)
    else:
        return float(text)