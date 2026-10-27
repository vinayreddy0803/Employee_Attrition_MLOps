import sys


def test_python_environment():
    assert sys.version_info.major == 3


def test_project_name():
    assert "Employee_Attrition_MLOps" == "Employee_Attrition_MLOps"
