import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

books_list = []

print("Starting to scrape books...")
print("Please wait, it will take a few minutes")

rating_dict = {}
rating_dict["One"] = 1
rating_dict["Two"] = 2
rating_dict["Three"] = 3
rating_dict["Four"] = 4
rating_dict["Five"] = 5

my_headers = {}
my_headers["User-Agent"] = "Mozilla/5.0"

page_number = 1
while page_number <= 50:
    my_link = "https://books.toscrape.com/catalogue/page-" + str(page_number) + ".html"

    try:
        my_page = requests.get(my_link, headers=my_headers, timeout=10)
    except:
        print("Internet problem on page", page_number, "trying again")
        time.sleep(3)
        try:
            my_page = requests.get(my_link, headers=my_headers, timeout=10)
        except:
            print("Still problem, skipping page", page_number)
            page_number = page_number + 1
            continue

    my_soup = BeautifulSoup(my_page.text, "html.parser")
    books_on_page = my_soup.find_all("article", class_="product_pod")

    for one_book in books_on_page:
        title = one_book.h3.a["title"]

        price_tag = one_book.find("p", class_="price_color")
        price_text = price_tag.text
        price_text = price_text.replace("£", "")
        price_text = price_text.replace("Â", "")
        price_text = price_text.strip()
        price = float(price_text)

        rating_tag = one_book.find("p", class_="star-rating")
        rating_word = rating_tag["class"][1]
        rating = rating_dict[rating_word]

        stock_tag = one_book.find("p", class_="instock availability")
        stock_text = stock_tag.text
        stock_text = stock_text.strip()
        if "In stock" in stock_text:
            in_stock = True
        else:
            in_stock = False
        try:
            book_link = one_book.h3.a["href"]
            full_book_link = "https://books.toscrape.com/catalogue/" + book_link
            book_page = requests.get(full_book_link, headers=my_headers, timeout=10)
            book_soup = BeautifulSoup(book_page.text, "html.parser")
            breadcrumb = book_soup.find("ul", class_="breadcrumb")
            breadcrumb_items = breadcrumb.find_all("li")
            category = breadcrumb_items[2].text
            category = category.strip()
            time.sleep(0.3)
        except:
            print("Could not get category for", title, "- putting Unknown")
            category = "Unknown"

        one_book_data = {}
        one_book_data["title"] = title
        one_book_data["price"] = price
        one_book_data["rating"] = rating
        one_book_data["rating_word"] = rating_word
        one_book_data["in_stock"] = in_stock
        one_book_data["category"] = category

        books_list.append(one_book_data)

    print("Finished page", page_number, "Total books now:", len(books_list))
    page_number = page_number + 1

df = pd.DataFrame(books_list)
df = df.dropna()
df = df.drop_duplicates()
print("Total books after cleaning:", len(df))
print(df.head())
df.to_csv("books_dataset.csv", index=False)
print("Saved csv file")

plt.figure(figsize=(8, 5))
sns.barplot(x="rating", y="price", data=df)
plt.title("Average Price by Rating")
plt.savefig("price_by_rating.png")
plt.close()
print("Made graph 1")

category_price = df.groupby("category")["price"].mean()
category_price = category_price.sort_values(ascending=False)
top_10 = category_price.head(10)
plt.figure(figsize=(10, 5))
top_10.plot(kind="bar")
plt.title("Average Price by Category - Top 10")
plt.savefig("price_by_category.png")
plt.close()
print("Made graph 2")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=20)
plt.title("Price Distribution")
plt.savefig("price_distribution.png")
plt.close()
print("Made graph 3")

plt.figure(figsize=(8, 5))
sns.countplot(x="rating", data=df)
plt.title("Number of Books by Rating")
plt.savefig("rating_count.png")
plt.close()
print("Made graph 4")

X = df[["rating"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
my_model = LinearRegression()
my_model.fit(X_train, y_train)
y_pred = my_model.predict(X_test)
my_mse = mean_squared_error(y_test, y_pred)
my_r2 = r2_score(y_test, y_pred)
print("MSE:", my_mse)
print("R2 Score:", my_r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.savefig("actual_vs_predicted.png")
plt.close()
print("Made graph 5")

print("ALL DONE")