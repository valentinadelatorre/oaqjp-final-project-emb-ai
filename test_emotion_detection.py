import unittest
from unittest.mock import patch, MagicMock
import json
from EmotionDetection import emotion_detector


def make_mock_response(anger, disgust, fear, joy, sadness):
    """Helper to build a mock Watson API response."""
    data = {
        "emotionPredictions": [{
            "emotion": {
                "anger": anger,
                "disgust": disgust,
                "fear": fear,
                "joy": joy,
                "sadness": sadness
            }
        }]
    }
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(data)
    return mock_resp


class TestEmotionDetector(unittest.TestCase):

    @patch('requests.post')
    def test_joy_dominant(self, mock_post):
        mock_post.return_value = make_mock_response(
            anger=0.005, disgust=0.002, fear=0.007, joy=0.960, sadness=0.026
        )
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result['dominant_emotion'], 'joy')

    @patch('requests.post')
    def test_anger_dominant(self, mock_post):
        mock_post.return_value = make_mock_response(
            anger=0.810, disgust=0.100, fear=0.040, joy=0.010, sadness=0.040
        )
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result['dominant_emotion'], 'anger')

    @patch('requests.post')
    def test_disgust_dominant(self, mock_post):
        mock_post.return_value = make_mock_response(
            anger=0.080, disgust=0.780, fear=0.050, joy=0.010, sadness=0.080
        )
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result['dominant_emotion'], 'disgust')

    @patch('requests.post')
    def test_sadness_dominant(self, mock_post):
        mock_post.return_value = make_mock_response(
            anger=0.030, disgust=0.020, fear=0.060, joy=0.015, sadness=0.875
        )
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result['dominant_emotion'], 'sadness')

    @patch('requests.post')
    def test_fear_dominant(self, mock_post):
        mock_post.return_value = make_mock_response(
            anger=0.040, disgust=0.030, fear=0.850, joy=0.010, sadness=0.070
        )
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
