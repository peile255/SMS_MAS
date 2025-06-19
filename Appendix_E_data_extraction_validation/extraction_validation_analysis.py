import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau


def generate_validation_analysis():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load the validation sample data
    df = pd.read_csv("extraction_validation_sample.csv")

    # Calculate basic agreement statistics
    total_fields = len(df)
    full_agreement = len(df[df['Resolution'] == 'Full'])
    resolved_discrepancies = len(df[df['Resolution'] == 'Resolved'])
    agreement_rate = full_agreement / total_fields

    # Field-specific agreement rates
    field_agreement = df.groupby('Field')['Resolution'].apply(
        lambda x: (x == 'Full').mean()
    ).reset_index(name='AgreementRate')

    # Paper-specific agreement rates
    paper_agreement = df.groupby('ID')['Resolution'].apply(
        lambda x: (x == 'Full').mean()
    ).reset_index(name='AgreementRate')

     # Also generate a markdown version for reference
    markdown_content = f"""## Extraction Validation Analysis

### Key Metrics
- Overall agreement rate: {agreement_rate:.0%}
- Fields requiring resolution: {resolved_discrepancies}
- Most reliable field: {field_agreement.loc[field_agreement['AgreementRate'].idxmax(), 'Field']} ({field_agreement['AgreementRate'].max():.0%})
- Least reliable field: {field_agreement.loc[field_agreement['AgreementRate'].idxmin(), 'Field']} ({field_agreement['AgreementRate'].min():.0%})

### Field Agreement Rates
| Field | Agreement Rate |
|-------|----------------|"""
    for _, row in field_agreement.iterrows():
        markdown_content += f"\n| {row['Field']} | {row['AgreementRate']:.0%} |"

    with open("results/extraction_validation_analysis.md", "w") as f:
        f.write(markdown_content)

    # Save the agreement rate tables as CSV for reference
    field_agreement.to_csv("results/field_agreement_rates.csv", index=False)
    paper_agreement.to_csv("results/paper_agreement_rates.csv", index=False)


if __name__ == "__main__":
    analysis = generate_validation_analysis()
    print("All files generated successfully in the 'results' directory:")
    print("- extraction_validation_analysis.md (Markdown version)")
    print("- field_agreement_rates.csv (Field agreement rates)")
    print("- paper_agreement_rates.csv (Paper agreement rates)")