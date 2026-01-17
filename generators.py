import random

class CaseGenerator:
    """
    Responsible for generating case-level attributes based on 
    domain-specific statistical distributions BPI 2017.
    According to data_validation.ipynb
    """
    def __init__(self):
        self.app_types = ["New credit", "Limit raise"]
        self.app_weights = [28120, 3389]
        
        self.goal_options = [
            "Car", "Home improvement", "Existing loan takeover",
            "Other, see explanation", "Unknown", "Not speficied",
            "Remaining debt home", "Extra spending limit", "Caravan / Camper",
            "Motorcycle", "Boat", "Tax payments", "Business goal", "Debt restructuring"
        ]
        self.goal_weights = [
            9328, 7669, 5601,
            2985, 2365, 1065,
            842, 625, 369,
            275, 201, 152, 30, 2
        ]

    def generate_new_case_attributes(self):
        """
        Returns a dictionary of attributes for a fresh case.
        """
        # 1. Application Type
        app_type = random.choices(self.app_types, weights=self.app_weights, k=1)[0]

        # 2. Loan Goal
        loan_goal = random.choices(self.goal_options, weights=self.goal_weights, k=1)[0]

        # 3. Requested Amount (Triangular distribution)
        # min=100, max=60000, mode=12500
        amount = round(random.triangular(100, 60000, 12500), 2)

        return {
            "case:ApplicationType": app_type,
            "case:LoanGoal": loan_goal,
            "case:RequestedAmount": amount
            # Add other attributes here easily in the future
        }