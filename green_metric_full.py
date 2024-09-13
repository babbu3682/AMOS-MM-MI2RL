import re
from green_score import GREEN

# Example reports
# pred_report = {"chest": "Example text", "abdomen": "Example text", "pelvis": "Example text"}
# gt_report = {"chest": "Example text", "abdomen": "Example text", "pelvis": "Example text"}

def split_report_into_sections(report):
    if isinstance(report, list):
        report = report[0]

    # 정규 표현식을 사용하여 [Chest], [Abdomen], [Pelvis]로 구분된 텍스트를 추출합니다.
    chest_match   = re.search(r'\[Chest\]:\s*(.*?)(?=\[Abdomen\]|\Z)', report, re.DOTALL)
    abdomen_match = re.search(r'\[Abdomen\]:\s*(.*?)(?=\[Pelvis\]|\Z)', report, re.DOTALL)
    pelvis_match  = re.search(r'\[Pelvis\]:\s*(.*)', report, re.DOTALL)
    
    # 각 부분이 존재하면 해당 텍스트를 사용하고, 그렇지 않으면 빈 문자열로 설정합니다.
    chest   = chest_match.group(1).strip() if chest_match else ''
    abdomen = abdomen_match.group(1).strip() if abdomen_match else ''
    pelvis  = pelvis_match.group(1).strip() if pelvis_match else ''
    
    return chest, abdomen, pelvis

def validate_and_format_reports(pred_report, gt_report):
    # 텍스트를 섹션으로 나눕니다
    pred_chest, pred_abdomen, pred_pelvis = split_report_into_sections(pred_report)
    gt_chest, gt_abdomen, gt_pelvis = split_report_into_sections(gt_report)

    # 각 키에 맞게 딕셔너리로 변환
    validated_pred_report = {
        "chest": pred_chest,
        "abdomen": pred_abdomen,
        "pelvis": pred_pelvis
    }

    validated_gt_report = {
        "chest": gt_chest,
        "abdomen": gt_abdomen,
        "pelvis": gt_pelvis
    }

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

    # Initialize list to store scores for non-empty GT parts
    green_scores = []

    # Calculate the GREEN score for each part, ignoring empty GT parts
    for part in gt_report:
        if gt_report[part].strip():  # Check if GT part is not empty
            # Calculate score for the current part
            score, greens, explanations = model(refs=[gt_report[part]], hyps=[pred_report[part]])
            green_scores.append(score)

    # Calculate the mean GREEN score 
    if green_scores:
        mean_green_score = sum(green_scores) / len(green_scores)
    else:
        mean_green_score = 0  # Handle case where no scores are calculated

    # Return the average score
    return mean_green_score
