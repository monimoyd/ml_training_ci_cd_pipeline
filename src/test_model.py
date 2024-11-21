import unittest
import sys
import torch
from src.model import MNISTModel
from src.utils import count_parameters, validate_model
sys.path.append('..\src')

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
    
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 25000, "Model has too many parameters")
    
    def test_input_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        try:
            output = self.model(test_input)
            self.assertEqual(output.shape[1], 10, "Output shape is incorrect")
        except Exception as e:
            self.fail(f"Model failed to process 28x28 input: {str(e)}")
    
    def test_model_accuracy(self):
        accuracy = validate_model('latest_model.pth')
        self.assertGreater(accuracy, 0.95, "Model accuracy is below 95%")

if __name__ == '__main__':
    unittest.main()