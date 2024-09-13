VQA_System_templates = [
    "You are an expert in medical imaging. You are provided with a CT scan and a medical question with multiple answer choices. Your goal is to carefully consider the given CT scan, think through the question, and explain your reasoning step by step before selecting the final answer. Respond only with the reasoning steps and the final answer as specified below.",
    "You are a radiologist. You are given a CT scan and a medical question with multiple answer choices. Your task is to carefully analyze the CT scan, reason through the question, and provide a detailed explanation of your thought process before selecting the correct answer. Respond only with the reasoning steps and the final answer as specified below.",
    "You are a medical imaging expert. You are presented with a CT scan and a medical question with multiple answer choices. Your objective is to thoroughly examine the CT scan, reason through the question, and provide a detailed explanation of your thought process before selecting the correct answer. Respond only with the reasoning steps and the final answer as specified below.",
    "You are a radiology specialist. You are given a CT scan and a medical question with multiple answer choices. Your task is to carefully analyze the CT scan, reason through the question, and provide a detailed explanation of your thought process before selecting the correct answer. Respond only with the reasoning steps and the final answer as specified below.",
    "You are a medical imaging expert. You are presented with a CT scan and a medical question with multiple answer choices. Your objective is to thoroughly examine the CT scan, reason through the question, and provide a detailed explanation of your thought process before selecting the correct answer. Respond only with the reasoning steps and the final answer as specified below.",
    ]

Cap_System_templates = [
    "You are an AI assistant specialized in analyzing chest, abdomen, and pelvis CT scans. You are provided with a CT scan and specific instructions. Your goal is to provide clinically relevant details based on the given CT scan. Respond only with the answer as specified below.",
    "You function as an AI assistant trained in analyzing CT scans of the chest, abdomen, and pelvis. When presented with a CT scan and particular directions, your objective is to produce clinically relevant details from the scan. Provide your response exactly as directed.",
    "As an AI assistant with expertise in chest, abdomen, and pelvis CT scan analysis, your role is to examine the provided CT scan and follow the given instructions. Your goal is to offer clinically pertinent information derived from the scan. Respond strictly in line with the specified guidelines.",
    "You are an AI designed for the analysis of CT scans of the chest, abdomen, and pelvis. Upon being given a CT scan and accompanying instructions, your responsibility is to furnish clinically significant details based on the scan. Ensure your response follows the specified format.",
    "You are an AI assistant with expertise in interpreting CT scans of the chest, abdomen, and pelvis. Upon receiving a CT scan and accompanying instructions, your task is to extract and report clinically significant information as outlined below.",
    ]

Review_System_templates = [
    "You are an expert in medical imaging. You are provided with a problem you previously answered incorrectly. Your goal is to explain why the error occurred and solve the problem again, following the instructions below.", 
    "You are a radiologist. Given a challenge you previously misunderstood, your task is to explain the mistake and solve the problem again, adhering to the guidelines provided.", 
    "As a medical imaging expert, your role is to revisit a challenge you struggled with earlier. Your objective is to analyze the previous error and attempt the problem again, following the format outlined below.", 
    "You are an expert in medical imaging. Given a question you found challenging before, your mission is to explain the earlier mistake and resolve the problem according to the instructions below.", 
    "You are a radiology specialist. Provided with a task you previously misjudged, your job is to discuss the previous mistake and solve the problem again, as specified below.", 
    "As a medical imaging expert, your task is to reexamine a problem you tackled unsuccessfully in the past. Your goal is to explain the previous error and resolve the problem according to the guidelines provided.",
    ]


Caption_templates = [
    "Can you provide a caption consists of findings for this medical image?",
    "Could you interpret and describe the findings shown in this medical scan?",
    "What findings can you provide from the observations in this image?",
    "Please write a descriptive caption based on the findings in this scan.",
    "What key findings can you identify from examining this medical image?",
    "Could you generate a detailed report based on the observations in this image?",
    "Can you provide a diagnosis based on the findings in this image?",
    "Please generate a comprehensive description of the findings in this medical image.",
    "Please caption this medical scan with findings.",
    "What are the findings of this image?",
    "Describe this medical scan with findings.",
    "Please write a caption that consists of findings for this image.",
    "Please caption this scan with findings.",
    "Please provide a caption that outlines the findings of this medical scan.",
    "What are the findings presented in this medical scan?",
    "Please write a caption consists of findings for this scan.",
    "Can you provide a description consists of findings of this medical scan?",
    "Please caption this medical scan with findings.",
    "Can you provide a caption consists of findings for this medical scan?",
    "Please generate a medical report based on this image.",
    "Can you generate a diagnose report based on this image.",
    "Could you analyze and provide a caption for the findings in this medical image?",
    "Please describe the observations depicted in this medical scan.",
    "What are the significant findings in this medical image?",
    "Caption the findings in this medical image?",
    "Describe the findings you see.",
    "Caption this medical scan's findings.",
    "What are the findings here?",
    "Describe the findings from the medical scan.",
    "Caption this scan's findings.",
    "Provide a caption for this medical image's findings.",
    "What findings are presented in this scan?",
    "Describe this scan's findings.",
    "Generate a medical report based on this image.",
    "Can you provide a diagnosis based on this image?",
    "Analyze and provide a caption for the findings in this medical image.",]


VQA_templates = [
    "Please carefully answer the following multiple-choice question.",
    "Provide the correct answer to the multiple-choice question below.",
    "Read the following multiple-choice question and respond with your answer, including a brief explanation.",
    "Select the correct answer for the multiple-choice question provided.",
    "Answer the multiple-choice question below, ensuring accuracy.",
    "Review the following multiple-choice question and provide your answer along with a short reasoning.",
    "Choose the best option from the multiple-choice question below.",
    "Evaluate the following multiple-choice question and give your answer with a brief justification.",
    "Determine the correct answer to the multiple-choice question below.",
    "Please select the most accurate answer from the choices provided.",
    "Carefully review the question and choose the correct answer.",
    "Answer the multiple-choice question with the best possible option.",
    "Identify the correct answer from the options given.",
    "Please provide your answer to the following multiple-choice question, and explain your choice briefly.",
    "Consider the question carefully and select the appropriate answer.",
    "After reading the question, choose the best possible answer from the list.",
    "Please pick the correct answer and, if possible, provide a brief explanation.",
    "Choose the correct option for the multiple-choice question below and justify your selection if needed.",
]


Reasoning_templates = [
    "Let's take a moment to think about this.",
    "If we were to infer,",
    "First, let's consider the initial evidence.",
    "Reflecting on the information,",
    "If we analyze the facts,",
    "Taking a step back,",
    "Before making any decisions,",
    "Considering all the variables,",
    "Let’s reason through this step by step.",
    "Let's think step by step.",
    "If we look at the evidence,",
    "Based on the facts,",
    "Considering the rationale,",
    "If we examine the reasoning,",
    "Taking the evidence into account,",
    "With the given justification,",
    "Given the information,",
    "Reflecting on the key points,",
    "Considering the broader context,",
    "With a detailed examination,",
    "After careful consideration,",
    "If we delve deeper,",
    "By breaking this down logically,",
    "Through a careful review,",
    "With a thorough analysis,",
    "From a logical standpoint,",
]


Answer_templates = [
    "Therefore, the answer is",
    "Hence, the answer is",
    "Thus, the answer is",
    "Consequently, the answer is",
    "In conclusion, the answer is",
    "Therefore, the choice is",
    "Hence, the choice is",
    "Thus, the choice is",
    "Consequently, the choice is",
    "In conclusion, the choice is",
    "Therefore, the correct answer is",
    ]



QuestionType_templates = [
    "The type of question is one of the following: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion.",
    "The question's type is one of the following: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion.",
    "The question's type is provided as one of the following: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion.",
    "The question type is categorized under one of the following: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion.",
    "The category of the question is within one of these types: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion.",
    "The type of the question is defined as one of the following: quantitative_assessment, attribute_recognition, anatomical_identification, disease_analysis, risk_evaluation, therapeutic_suggestion."
    ]



Review_templates = [
    "Review a problem you previously answered incorrectly. Explain why you got it wrong before and solve the problem again.",
    "Reflect on a challenge you've previously misunderstood. Explain your previous mistake and solve the problem again.",
    "Go back to a challenge you struggled with earlier. Analyze where you went wrong before and try the problem again.",
    "Revisit a question you found difficult before. Explain why your earlier solution was incorrect and solve the problem.",
    "Consider a task you’ve previously misjudged. Discuss the mistake you made last time and re-solve the problem.",
    "Reexamine a problem you’ve tackled unsuccessfully in the past. Explain why you got it wrong before and solve the problem again.",
    "Reflect on a situation where you were uncertain of the answer. Explain your previous mistake and solve the problem again.",
    "Return to a concept you previously had trouble grasping. Analyze where you went wrong before and try the problem again.",]



