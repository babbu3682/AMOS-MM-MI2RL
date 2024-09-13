import re
from green_score import GREEN

# Example reports
# pred_report = {"findings": "Example text"}
# gt_report = {"findings": "Example text"}


def validate_and_format_reports(pred_report, gt_report):
    if isinstance(pred_report, list):
        pred_report = pred_report[0]

    if isinstance(gt_report, list):
        gt_report = gt_report[0]

    # 각 키에 맞게 딕셔너리로 변환
    validated_pred_report = {"findings": pred_report}

    validated_gt_report = {"findings": gt_report}

    return validated_pred_report, validated_gt_report

def calculate_mean_green_score(pred_report, gt_report):
    # Validate and format the reports
    pred_report, gt_report = validate_and_format_reports(pred_report, gt_report)
    
    # Initialize the GREEN model
    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        do_sample=False,  # should be always False
        batch_size=16,
        return_0_if_no_green_score=True,
        cuda=True
    )

    score = None
    explanations = None
    # Calculate the GREEN score for each part, ignoring empty GT parts
    if gt_report['findings'].strip():  # Check if GT part is not empty
        # Calculate score for the current part
        score, _, explanations = model(refs=[gt_report['findings']], hyps=[pred_report['findings']])

    # Calculate the mean GREEN score 
    if score is not None:
        green_score = score
    else:
        green_score = 0  # Handle case where no scores are calculated

    # Return the average score
    return green_score, explanations
