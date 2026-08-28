import pandas as pd
import pickle


# Load trained model
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)


print("Student Placement Prediction")
print("-----------------------------")

cgpa = float(input("Enter CGPA: "))
attendance = float(input("Enter Attendance: "))
coding_score = float(input("Enter Coding Score: "))
projects = int(input("Enter Number of Projects: "))
internship = int(input("Internship (1 = Yes, 0 = No): "))


student = pd.DataFrame([[
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
]], columns=[
    "CGPA",
    "Attendance",
    "CodingScore",
    "Projects",
    "Internship"
])


prediction = model.predict(student)[0]


if prediction == 1:
    print("Predicted Result: PLACED")
else:
    print("Predicted Result: NOT PLACED")