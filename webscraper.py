import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# List of URLs to scrape
urls = [
    "https://markets.businessinsider.com/bonds/finder?p=1&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19",
    "https://markets.businessinsider.com/bonds/finder?p=2&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19",
    "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
]

# Define headers for the HTTP request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
}

# Function to scrape bond data from a single URL
def scrape_bond_data(url, headers):
    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch page at {url}. Status code: {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the bond data table (replace the class with the actual one from the webpage)
    bond_table = soup.find("table", {"class": "table"})  # Update the class name
    if not bond_table:
        print(f"Bond table not found at {url}")
        return None

    # Extract headers
    headers = [th.text.strip() for th in bond_table.find_all("th")]

    # Extract rows
    rows = bond_table.find_all("tr")
    data = []
    for row in rows[1:]:  # Skip header row
        cells = row.find_all("td")
        data.append([cell.text.strip() for cell in cells])

    # Return data as a DataFrame
    return pd.DataFrame(data, columns=headers)

# Loop through URLs and scrape data
all_data = []
for url in urls:
    print(f"Scraping data from {url}...")
    df = scrape_bond_data(url, headers)
    if df is not None:
        # Add source URL and scrape date to the data
        df['Source URL'] = url
        df['Scrape Date'] = datetime.now().strftime("%Y-%m-%d")
        all_data.append(df)

# Combine all data into a single DataFrame
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    # Append data to the existing file or create a new one
    file_name = "bond_data_combined.csv"
    try:
        existing_data = pd.read_csv(file_name)
        final_df = pd.concat([existing_data, final_df], ignore_index=True)
    except FileNotFoundError:
        print("File not found. Creating a new one.")

    final_df.to_csv(file_name, index=False)
    print(f"Data appended to {file_name}")
else:
    print("No data was scraped.")
