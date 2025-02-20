
import tensorflow as tf
import tensorflow_transform as tft
    
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}

    attendance = inputs['Attendance (%)']
    outputs['normalized_attendance'] = tft.scale_to_z_score(attendance)

    grades = inputs['Grades']
    outputs['normalized_grades'] = tft.scale_to_0_1(grades)

    sleep_hours = inputs['Sleep Hours']
    outputs['normalized_sleep_hours'] = tft.scale_to_0_1(sleep_hours)

    socioeconomic_score = inputs['Socioeconomic Score']
    outputs['normalized_socioeconomic_score'] = tft.scale_to_0_1(socioeconomic_score)

    study_hours = inputs['Study Hours']
    outputs['normalized_study_hours'] = tft.scale_to_0_1(study_hours)
    
    return outputs
