import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton
from sklearn.decomposition import PCA
from pathlib import Path


SELECTED_ISINS = [
    "CA135087K528", "CA135087K940", 
    "CA135087L518", "CA135087L930",   
    "CA135087M847", "CA135087N837", 
    "CA135087P576", "CA135087Q491",   
    "CA135087Q988", "CA135087R895", 
    "CA135087S471"   
]



def clean_and_prepare_data(df):
    # Clean data
    df = df.copy()
    
    # Uppercase ISINs and remove whitespace
    df['ISIN'] = df['ISIN'].str.upper().str.strip()
    
    # Clean other columns - remove whitespace
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Filter and deduplicate
    df = df[df['ISIN'].isin(SELECTED_ISINS)]
    df = df.drop_duplicates(subset=['ISIN', 'Scrape Date'])
    
    # Convert dates
    date_formats = ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"]
    date_columns = ['Maturity Date', 'Issue Date', 'Scrape Date']
    
    for col in date_columns:
        for fmt in date_formats:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt)
                break
            except (ValueError, TypeError):
                continue
    
    # Remove weekend dates
    df = df[df['Scrape Date'].dt.day_name().isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    
    # Convert Coupon to numeric
    df['Coupon'] = df['Coupon'].str.replace('%', '').astype(float) / 100 
    
    # Process numeric columns
    df['Bid'] = pd.to_numeric(df['Bid'], errors='coerce')
    df['Ask'] = pd.to_numeric(df['Ask'], errors='coerce')
    df['Price'] = (df['Bid'] + df['Ask']) / 2
    df['Price'] = df['Price'].astype(float)
    
    # Calculate Years to Maturity
    df['Years to Maturity'] = (df['Maturity Date'] - df['Scrape Date']).dt.days / 365.25
    
    return df

def verify_and_count_bonds(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean and prepare the data
    cleaned_df = clean_and_prepare_data(df)
    
    # Group by Scrape Date and count bonds, keeping only dates with exactly 10 bonds
    bond_counts = cleaned_df.groupby('Scrape Date')['ISIN'].count()
    valid_dates = bond_counts[bond_counts == 11]
    
    print("Verified Dates with Exactly 10 Bonds:")
    for date, count in bond_counts.items():
        print(f"{date.date()}: {count} bonds")
    
    # Filter the dataframe to include only these dates
    filtered_df = cleaned_df[cleaned_df['Scrape Date'].isin(valid_dates.index)]
    
    return filtered_df

# Example usage
# verify_and_count_bonds('APM466Data- Combined with ISIN and issue date.csv')

def calculate_ytm(price, coupon, years_to_maturity, maturity_date, reference_date, face_value=100):
    if years_to_maturity <= 0.5:
        return ((face_value / price) ** (1 / years_to_maturity) - 1) * 100
    
    coupon_payment = coupon * face_value
    payment_dates = []
    current_date = maturity_date
    
    while current_date > reference_date:
        payment_dates.append(current_date)
        current_date -= pd.DateOffset(months=6)  # payment schedule
    
    payment_dates = sorted(payment_dates)
    times_to_payments = [(date - reference_date).days / 365.25 for date in payment_dates]
    
    def ytm_function(y):
        return sum([coupon_payment * np.exp(-y * t) for t in times_to_payments[:-1]]) + \
               (coupon_payment + face_value) * np.exp(-y * times_to_payments[-1]) - price
    
    initial_guess = 0.03 if price > face_value else 0.08
    try:
        ytm = newton(ytm_function, initial_guess)
    except RuntimeError:
        ytm = np.nan
    
    return ytm * 100

def plot_yield_curves(df):
    plt.figure(figsize=(10, 6))
    
    # Group by Scrape Date
    grouped = df.groupby('Scrape Date')
    
    for reference_date, group in grouped:
        # Sort and calculate YTM
        group = group.sort_values(by='Years to Maturity')
        group['YTM'] = group.apply(
            lambda row: calculate_ytm(
                row['Price'], 
                row['Coupon'], 
                row['Years to Maturity'], 
                row['Maturity Date'], 
                reference_date
            ), 
            axis=1
        )
        
        # Add origin point
        group = pd.concat([
            pd.DataFrame({'Years to Maturity': [0], 'YTM': [0]}), 
            group
        ], ignore_index=True)
        
        # Interpolate yield curve
        max_years = group['Years to Maturity'].max()
        interpolation = interp1d(
            group['Years to Maturity'], 
            group['YTM'], 
            kind='linear', 
            fill_value='extrapolate'
        )
        years = np.linspace(0, max_years, 100)
        
        # Plot the interpolated yield curve
        plt.plot(
            years, 
            interpolation(years), 
            label=str(reference_date.date())
        )
    
    # Finalize plot
    plt.title('Yield Curves')
    plt.xlabel('Years to Maturity')
    plt.ylabel('Yield to Maturity (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def get_payment_schedule(maturity_date, reference_date):
    payment_dates = []
    current_date = maturity_date
    while current_date > reference_date:
        payment_dates.append(current_date)
        current_date -= pd.DateOffset(months=6)  # payment schedule
    payment_dates = sorted(payment_dates)
    times_to_payments = [(date - reference_date).days / 365.25 for date in payment_dates]
    return times_to_payments

def bootstrap_spot_rates(df, face_value=100):
    df = df.sort_values(by='Years to Maturity')
    spot_rates = []
    
    for _, row in df.iterrows():
        price = row['Price']
        coupon = row['Coupon'] * face_value
        maturity_date = row['Maturity Date']
        reference_date = row['Scrape Date']  # Changed from 'Reference Date'
        
        times_to_payments = get_payment_schedule(maturity_date, reference_date)
        
        if len(spot_rates) == 0:
            # zcb
            spot_rate = -np.log(price / face_value) / times_to_payments[-1]
        else:
            # bootstrapping
            def spot_function(r):
                return sum([coupon / (1 + s/2)**(2*t) for s, t in zip(spot_rates, times_to_payments[:-1])]) + \
                    (coupon + face_value) / (1 + r/2)**(2*times_to_payments[-1]) - price
            
            spot_rate = newton(spot_function, 0.05)
        
        spot_rates.append(spot_rate)
    
    df['Spot Rate'] = [rate * 100 for rate in spot_rates]
    return df

def plot_spot_curves(df):
    plt.figure(figsize=(10, 6))
    
    # Group by Scrape Date
    grouped = df.groupby('Scrape Date')
    
    for reference_date, group in grouped:
        group = bootstrap_spot_rates(group)
        plt.plot(
            group['Years to Maturity'], 
            group['Spot Rate'], 
            label=str(reference_date.date())
        )
    
    plt.title('Spot Curves')
    plt.xlabel('Years to Maturity')
    plt.ylabel('Spot Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def calculate_forward_rates(df):
    df = bootstrap_spot_rates(df)

    forward_rates = []
    spot_rates = df['Spot Rate'].values / 100 
    maturities = df['Years to Maturity'].values

    for i in range(1, len(maturities)):
        t1 = maturities[i - 1]
        t2 = maturities[i]
        s1 = spot_rates[i - 1]
        s2 = spot_rates[i]

        # 1-year forward rate from t1 to t2
        forward_rate = ((1 + s2)**t2 / (1 + s1)**t1)**(1 / (t2 - t1)) - 1
        forward_rates.append(forward_rate * 100) 

    df['Forward Rate'] = [np.nan] + forward_rates
    return df

def plot_forward_curves(df):
    plt.figure(figsize=(10, 6))
    
    # Group by Scrape Date
    grouped = df.groupby('Scrape Date')
    
    for reference_date, group in grouped:
        group = calculate_forward_rates(group)
        plt.plot(
            group['Years to Maturity'][1:], 
            group['Forward Rate'][1:], 
            label=str(reference_date.date())
        )
    
    plt.title('1-Year Forward Curves')
    plt.xlabel('Years to Maturity')
    plt.ylabel('Forward Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def calculate_log_returns(values):
    return np.log(values[1:] / values[:-1])

def perform_covariance_and_pca_analysis(df, output_path):
    # Group by Scrape Date to ensure chronological order
    grouped = df.groupby('Scrape Date').apply(lambda x: x.sort_values('Years to Maturity')).reset_index(drop=True)
    
    # Extract yield and forward data
    yield_data = []
    forward_data = []
    
    # Group by unique dates
    for _, group in grouped.groupby('Scrape Date'):
        group = calculate_forward_rates(group)
        yield_data.append(group['Spot Rate'].values[::2])
        forward_data.append(group['Forward Rate'].values[::2])
    
    yield_array = np.array(yield_data).T 
    forward_array = np.array(forward_data).T
    
    # Calculate log returns
    yield_log_returns = np.apply_along_axis(calculate_log_returns, 1, yield_array)
    forward_log_returns = np.apply_along_axis(calculate_log_returns, 1, forward_array)
    
    # Handle potential NaN values
    yield_log_returns = np.nan_to_num(yield_log_returns)
    forward_log_returns = np.nan_to_num(forward_log_returns)
    
    # Compute covariance matrices
    yield_cov_matrix = np.cov(yield_log_returns, rowvar=True)
    forward_cov_matrix = np.cov(forward_log_returns, rowvar=True)
    
    # Perform PCA
    pca_yield = PCA()
    pca_forward = PCA()
    pca_yield.fit(yield_cov_matrix)
    pca_forward.fit(forward_cov_matrix)
    
    # Prepare results
    results = {
        "Yield Covariance Matrix": yield_cov_matrix,
        "Forward Covariance Matrix": forward_cov_matrix,
        "Yield PCA Eigenvalues": pca_yield.explained_variance_,
        "Yield PCA Eigenvectors": pca_yield.components_,
        "Forward PCA Eigenvalues": pca_forward.explained_variance_,
        "Forward PCA Eigenvectors": pca_forward.components_
    }
    
    # Save results
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        filename = f"{key.replace(' ', '_').lower()}.csv"
        pd.DataFrame(value).to_csv(Path(output_path) / filename, index=False)
    
    print("Covariance matrices and PCA analysis saved to:", output_path)
    
    return results

# def plot_yield_curves(df):
#     plt.figure(figsize=(10, 6))
    
#     # Group by Scrape Date
#     grouped = df.groupby('Scrape Date')
    
#     for reference_date, group in grouped:
#         group = group.sort_values(by='Years to Maturity')
        
#         # Calculate YTM for each bond
#         group['YTM'] = group.apply(lambda row: calculate_ytm(
#             row['Price'], 
#             row['Coupon'], 
#             row['Years to Maturity'], 
#             row['Maturity Date'], 
#             reference_date
#         ), axis=1)
        
#         # Add point at origin
#         group = pd.concat([pd.DataFrame({'Years to Maturity': [0], 'YTM': [0]}), group], ignore_index=True)
        
#         max_years = group['Years to Maturity'].max()
#         interpolation = interp1d(group['Years to Maturity'], group['YTM'], kind='linear', fill_value='extrapolate')
#         years = np.linspace(0, max_years, 100)
        
#         plt.plot(years, interpolation(years), label=str(reference_date.date()))
    
#     plt.title('Yield Curves')
#     plt.xlabel('Years to Maturity')
#     plt.ylabel('Yield to Maturity (%)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# Main execution
def main(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean and prepare the data
    cleaned_df = clean_and_prepare_data(df)
    
    verify_and_count_bonds(file_path)
    
    # Plot yield curves
    plot_yield_curves(cleaned_df)
    plot_spot_curves(cleaned_df)
    plot_forward_curves(cleaned_df)
    perform_covariance_and_pca_analysis(cleaned_df,output_path="analysis_results")

# Example usage
# main('your_data_file.csv')
if __name__ == "__main__":
    main('APM466Data- Combined with ISIN and issue date.csv')
