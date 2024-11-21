Steps to run locally:
1. Create a virtual environment:
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
2. Install dependencies:
Bash
pip install -r requirements.txt

3. Train the model:
Bash
python src/train.py

4.Run tests:
Bash
python -m unittest tests/test_model.py

5. To deploy to GitHub:
i. Create a new repository on GitHub
ii.Initialize local git repository:
Bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

The GitHub Actions workflow will automatically:
i. Set up a Python environment
ii.Install dependencies
iii.Train the model
iv. Run all tests
v.Save the trained model as an artifact
The tests check for:
i. Model parameter count (< 25000)
ii.Input shape compatibility (28x28)
iii.Model accuracy (> 95%)
The model file is saved with a timestamp suffix (e.g., model_20240321_143022.pth) for tracking when it was trained.
Note: You might need to adjust the accuracy threshold in the tests if the model doesn't achieve 95% accuracy in one epoch. You could either train for more epochs or lower the threshold for the CI/CD pipeline to pass.
