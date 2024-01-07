import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

class LoanAnalysis:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def calculate_recovery_percentage(self):
        total_funded = self.data_frame['funded_amnt'].sum()
        total_recovered = self.data_frame['total_rec_prncp'].sum()
        recovery_percentage = (total_recovered / total_funded) * 100
        return recovery_percentage, total_funded

    def visualize_recovery_percentage(self, future_months=6):
        current_date = datetime.now()
        future_date = current_date + timedelta(days=30 * future_months)

        filtered_data = self.data_frame[(self.data_frame['issue_d'] >= current_date) & (self.data_frame['issue_d'] <= future_date)]

        total_funded = filtered_data['funded_amnt'].sum()
        total_recovered = filtered_data['total_rec_prncp'].sum()
        recovery_percentage = (total_recovered / total_funded) * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Total Recovery', f'Recovery in {future_months} months'], y=[recovery_percentage, 100], palette="viridis")
        plt.title('Percentage of Loans Recovered')
        plt.ylabel('Percentage')
        plt.show()

if __name__ == "__main__":
    credentials = load_credentials()
    db_connector = RDSDatabaseConnector(credentials)

    data_frame = db_connector.extract_data_to_dataframe()

    loan_analysis = LoanAnalysis(data_frame)

    current_recovery_percentage, total_funded = loan_analysis.calculate_recovery_percentage()
    print(f"Current Recovery Percentage: {current_recovery_percentage:.2f}% of ${total_funded:,}")

    loan_analysis.visualize_recovery_percentage(future_months=6)

class LossAnalysis:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def calculate_loss_percentage(self):
        charged_off_loans = self.data_frame[self.data_frame['loan_status'] == 'Charged Off']
        total_loans = len(self.data_frame)
        total_charged_off = len(charged_off_loans)
        loss_percentage = (total_charged_off / total_loans) * 100
        return loss_percentage, charged_off_loans

    def calculate_total_loss_amount(self, charged_off_loans):
        total_loss_amount = charged_off_loans['total_rec_prncp'].sum()
        return total_loss_amount

if __name__ == "__main__":
    credentials = load_credentials()
    db_connector = RDSDatabaseConnector(credentials)

    data_frame = db_connector.extract_data_to_dataframe()

    loss_analysis = LossAnalysis(data_frame)

    loss_percentage, charged_off_loans = loss_analysis.calculate_loss_percentage()
    print(f"Percentage of Charged Off Loans: {loss_percentage:.2f}%")

    total_loss_amount = loss_analysis.calculate_total_loss_amount(charged_off_loans)
    print(f"Total Amount Paid towards Charged Off Loans: ${total_loss_amount:,.2f}")

class ProjectedLossAnalysis:
    def __init__(self, charged_off_loans):
        self.charged_off_loans = charged_off_loans

    def calculate_projected_loss(self):
        self.charged_off_loans['remaining_term'] = self.charged_off_loans['term'] - self.charged_off_loans['installment']
        self.charged_off_loans['projected_loss'] = self.charged_off_loans['remaining_term'] * self.charged_off_loans['installment']
        total_projected_loss = self.charged_off_loans['projected_loss'].sum()
        return total_projected_loss

    def visualize_projected_loss(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.charged_off_loans['remaining_term'], kde=True, bins=np.arange(0, max(self.charged_off_loans['remaining_term'])+1, 1))
        plt.title('Distribution of Remaining Term for Charged Off Loans')
        plt.xlabel('Remaining Term (Months)')
        plt.ylabel('Count')
        plt.show()

if __name__ == "__main__":
    credentials = load_credentials()
    db_connector = RDSDatabaseConnector(credentials)

    data_frame = db_connector.extract_data_to_dataframe()

    loss_analysis = LossAnalysis(data_frame)

    loss_percentage, charged_off_loans = loss_analysis.calculate_loss_percentage()
    print(f"Percentage of Charged Off Loans: {loss_percentage:.2f}%")

    total_loss_amount = loss_analysis.calculate_total_loss_amount(charged_off_loans)
    print(f"Total Amount Paid towards Charged Off Loans: ${total_loss_amount:,.2f}")

    projected_loss_analysis = ProjectedLossAnalysis(charged_off_loans)

    total_projected_loss = projected_loss_analysis.calculate_projected_loss()
    print(f"Projected Loss: ${total_projected_loss:,.2f}")

    projected_loss_analysis.visualize_projected_loss()

class LatePaymentsAnalysis:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def calculate_late_payments_percentage(self):
        late_payments = self.data_frame[self.data_frame['loan_status'].isin(['Late (16-30 days)', 'Late (31-120 days)'])]
        total_loans = len(self.data_frame)
        total_late_payments = len(late_payments)
        late_payments_percentage = (total_late_payments / total_loans) * 100
        return late_payments_percentage, late_payments

    def calculate_total_customers_and_loss(self, late_payments):
        total_customers = len(late_payments)
        total_loss_amount = late_payments['total_rec_prncp'].sum()
        return total_customers, total_loss_amount

    def calculate_projected_loss(self, late_payments):
        late_payments['remaining_term'] = late_payments['term'] - late_payments['installment']
        late_payments['projected_loss'] = late_payments['remaining_term'] * late_payments['installment']
        total_projected_loss = late_payments['projected_loss'].sum()
        return total_projected_loss

if __name__ == "__main__":
    credentials = load_credentials()
    db_connector = RDSDatabaseConnector(credentials)

    data_frame = db_connector.extract_data_to_dataframe()

    late_payments_analysis = LatePaymentsAnalysis(data_frame)

    late_payments_percentage, late_payments = late_payments_analysis.calculate_late_payments_percentage()
    print(f"Percentage of Customers with Late Payments: {late_payments_percentage:.2f}%")

    total_customers, total_loss_amount = late_payments_analysis.calculate_total_customers_and_loss(late_payments)
    print(f"Total Customers with Late Payments: {total_customers}")
    print(f"Total Loss Amount for Late Payments: ${total_loss_amount:,.2f}")

    total_projected_loss = late_payments_analysis.calculate_projected_loss(late_payments)
    print(f"Projected Loss for Late Payments: ${total_projected_loss:,.2f}")

    total_expected_revenue = data_frame['total_rec_prncp'].sum()
    late_payments_and_defaulted = late_payments_analysis.data_frame[late_payments_analysis.data_frame['loan_status'].isin(['Charged Off'])]
    total_loss_and_defaulted = late_payments_and_defaulted['total_rec_prncp'].sum()

    total_expected_revenue_percentage = (total_loss_and_defaulted / total_expected_revenue) * 100
    print(f"Percentage of Total Expected Revenue from Late Payments and Defaulted Loans: {total_expected_revenue_percentage:.2f}%")

class LoanIndicatorsAnalysis:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def create_subset(self):
        subset_users = self.data_frame[self.data_frame['loan_status'].isin(['Charged Off', 'Late (16-30 days)', 'Late (31-120 days)'])]
        return subset_users

    def visualize_indicators(self, column_name):
        plt.figure(figsize=(12, 6))
        sns.countplot(x=column_name, hue='loan_status', data=self.data_frame, order=self.data_frame[column_name].value_counts().index)
        plt.title(f'Loan Status Distribution by {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.show()

if __name__ == "__main__":
    credentials = load_credentials()
    db_connector = RDSDatabaseConnector(credentials)

    data_frame = db_connector.extract_data_to_dataframe()

    loan_indicators_analysis = LoanIndicatorsAnalysis(data_frame)

    subset_users = loan_indicators_analysis.create_subset()

    indicators_to_visualize = ['grade', 'purpose', 'home_ownership']

    for indicator in indicators_to_visualize:
        loan_indicators_analysis.visualize_indicators(indicator)
