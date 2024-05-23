import plotly.express as px

from sklearn import metrics


def get_attn_head_roc(
    ground_truth,
    data,
    task_name,
    visualize=True,
    additional_title="",
):
    fp, tp, thresh = metrics.roc_curve(ground_truth.flatten(), data.flatten())

    score = metrics.roc_auc_score(ground_truth.flatten(), data.flatten())

    if visualize:
        print("Score:", score)
        px.line(
            x=fp,
            y=tp,
            title=f"ROC Curve for {task_name} " + additional_title,
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        ).show()

    return (
        score,
        fp,
        tp,
        thresh,
    )
