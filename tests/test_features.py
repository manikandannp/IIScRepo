
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bike_rental_model.config.core import config
from bike_rental_model.processing.features import WeekdayEncoder


def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = WeekdayEncoder(
        variables='weekday', 
    )
    assert np.isnan(sample_input_data.loc[7046,'weekday'])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[7046,'weekday'] == 'Wed'