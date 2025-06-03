from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from st_aggrid import JsCode

def match_single_row(label, label_to_var):
    if not label_to_var:
        return ""
    labels = list(label_to_var.keys())
    vect = TfidfVectorizer().fit([label] + labels)
    vec1 = vect.transform([label])
    vec2 = vect.transform(labels)
    sim = cosine_similarity(vec1, vec2).flatten()
    idx = sim.argmax()
    return label_to_var[labels[idx]] if sim[idx] > 0.6 else ""

def get_cell_style_js():
    return JsCode("""
        function(params) {
            const main = params.data["Main Variable"];
            const val = params.value;
            const changed = params.data["label_changed"] === true;
            if (params.colDef.field === "Main Variable" || params.colDef.field === "Main Label") {
                return {backgroundColor: "#dee2e6", fontWeight: "bold"};
            }
            if (params.colDef.field === "Derivation") {
                return {backgroundColor: "#cfe2ff"};
            }
            if (!val) {
                return {backgroundColor: "#f8d7da"};
            }
            if (changed) {
                return {backgroundColor: "#fff3cd"};
            }
            if (val === main) {
                return {backgroundColor: "white"};
            }
            return {backgroundColor: "#fce5cd"};
        }
    """)


def detect_label_changes(old_df, new_df):
    """
    Detect which rows have changed 'Main Label' compared to old version.
    Returns list of row indices.
    """
    changed_rows = []
    for idx in range(len(old_df)):
        if idx >= len(new_df):
            continue
        old_label = str(old_df.at[idx, "Main Label"])
        new_label = str(new_df.at[idx, "Main Label"])
        if old_label.strip() != new_label.strip():
            changed_rows.append(idx)
    return changed_rows