# -*- coding: utf-8 -*-


from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import process
from flask_cors import CORS

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')



app = Flask(__name__)
CORS(app)
df = pd.read_csv('menu.csv', encoding='ISO-8859-1', delimiter=',')
df = df[['Id','Category', 'Item', 'Description', 'Price']]  
df = df.dropna()

@app.route('/api/sort_menu', methods=['POST'])
def sort_menu():
    data = request.get_json()

    # Extract the preferred items from the data
    preferred_items = data['preferred_items']

    # Create a new column that will be used for sorting
    df['sort'] = df['Item'].apply(lambda x: preferred_items.index(x) if x in preferred_items else len(preferred_items))

    # Sort the dataframe first by the new column, then by Category and Item
    df_sorted = df.sort_values(by=['sort', 'Category', 'Item'])

    # Drop the sort column as it's no longer needed
    df_sorted = df_sorted.drop(columns=['sort'])

    # Convert the sorted dataframe to a dictionary grouped by category
    sorted_menu = df_sorted.groupby('Category')['Item', 'Description', 'Price'].apply(list).to_dict()

    # Return the sorted menu as a JSON response
    return jsonify(sorted_menu)

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the menu data
vectorizer.fit(df['Item'])
 
"""Get synonyms for a given word using WordNet."""
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)



@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    # Get the food and drink names from the request data
    food_name = data['food']
    drink_name = data['drink']

    # Calculate the cosine similarity between the food and drink names and the menu items
   # food_vector = vectorizer.transform([food_name])
   # drink_vector = vectorizer.transform([drink_name])
    #food_similarities = cosine_similarity(food_vector, vectorizer.transform(df['Item']))
    #drink_similarities = cosine_similarity(drink_vector, vectorizer.transform(df['Item']))

    # Get the recommended food and drink items
    #recommended_food_indices = np.argsort(food_similarities[0])[::-1]  # Get the indices of the most similar food items
    #recommended_drink_indices = np.argsort(drink_similarities[0])[::-1]  # Get the indices of the most similar drink items

    # Filter the recommended items based on their category
    #recommended_foods = df.iloc[recommended_food_indices]
    #recommended_drinks = df.iloc[recommended_drink_indices]
    #recommended_foods = recommended_foods[~recommended_foods['Category'].isin(['Beverages', 'Coffee & Tea'])]['Item'].tolist()[:100]
    #recommended_drinks = recommended_drinks[recommended_drinks['Category'].isin(['Beverages', 'Coffee & Tea'])]['Item'].tolist()[:100]

    # Get the recommended food and drink items
    #recommended_foods = process.extractBests(food_name, df[df['Category'].isin(['Beverages', 'Coffee & Tea'])==False]['Item'].tolist(), limit=200)
    #recommended_drinks = process.extractBests(drink_name, df[df['Category'].isin(['Beverages', 'Coffee & Tea'])]['Item'].tolist(), limit=200)


    # Get synonyms
    food_synonyms = get_synonyms(food_name)
    drink_synonyms = get_synonyms(drink_name)


    # Separate the dataframe into foods and drinks
    food_df = df[df['Category'].isin(['Beverages', 'Coffee & Tea', 'Smoothies & Shakes']) == False]
    drink_df = df[df['Category'].isin(['Beverages', 'Coffee & Tea', 'Smoothies & Shakes'])]

    # Get the recommended food and drink items using fuzzy matching
    recommended_foods = []
    for term in [food_name] + food_synonyms:
        recommended_foods.extend(process.extractBests(term, food_df['Item'].tolist(), limit=100))

    recommended_drinks = []
    for term in [drink_name] + drink_synonyms:
        recommended_drinks.extend(process.extractBests(term, drink_df['Item'].tolist(), limit=100))

    print(food_df)

    # Extract the item names and their categories from the recommendations
    recommended_foods = [{"name": food[0], 
        "id": str(food_df[food_df['Item'] == food[0]]['Id'].values[0]),
        "category": food_df[food_df['Item'] == food[0]]['Category'].values[0],
        "description": food_df[food_df['Item'] == food[0]]['Description'].values[0], 
        "price": food_df[food_df['Item'] == food[0]]['Price'].values[0]} for food in recommended_foods]

    recommended_drinks = [{"name": drink[0],
        "id": str(drink_df[drink_df['Item'] == drink[0]]['Id'].values[0]),
        "category": drink_df[drink_df['Item'] == drink[0]]['Category'].values[0],
        "description": drink_df[drink_df['Item'] == drink[0]]['Description'].values[0], 
        "price": drink_df[drink_df['Item'] == drink[0]]['Price'].values[0]} for drink in recommended_drinks]


    return jsonify({
       'recommended_foods': recommended_foods,
        'recommended_drinks': recommended_drinks,
    })

@app.route('/api/menu', methods=['GET'])
def get_menu():
    menu = {'Foods': [], 'Drinks': []}
    for index, row in df.iterrows():
        category = row['Category']
        item = {"name": row['Item'], "description": row['Description'], "price": row['Price']}
        if category in ['Beverages', 'Coffee & Tea', 'Smoothies & Shakes']:
            menu['Drinks'].append(item)
        else:
            menu['Foods'].append(item)
    return jsonify(menu)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5008) 


('''

# Create an instance of the NLTK Porter Stemmer
stemmer = PorterStemmer()

# Define a function to process a category of food
def process_category(category):                                                 
    processed_category = []
    for item in category:
        # Make all text lowercase
        lower_case_item = item.lower()

        # Tokenize the string (split it into individual words)
        tokens = word_tokenize(lower_case_item)

        # Remove stopwords
        tokens = [token for token in tokens if token not in stopwords.words('english')]

        # Apply stemming to the tokens (reduce words to their base form)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        # Join the tokens back into a single string
        processed_category.append(" ".join(stemmed_tokens))  

    return processed_category



class RecommendationEngine:
    def __init__(self):
        self.dishes = {
            "ALCOHOLIC BEVERAGES": {
                "Mimosa": np.array([1, 0, 0, 1, 0]),
                "Caesar": np.array([0, 1, 1, 0, 1]),
                "Tequila Sunrise": np.array([0, 1, 1, 0, 1]),
                "Espresso Martini": np.array([0, 1, 1, 0, 1]),
                "Kentucky Mule": np.array([0, 1, 1, 0, 1]),
                "Strawberry Mojito": np.array([0, 1, 1, 0, 1]),
                "Grapefruit & T": np.array([0, 1, 1, 0, 1]),
                "House Sangria": np.array([0, 1, 1, 0, 1]),
                "B52": np.array([0, 1, 1, 0, 1]),
                "Irish Coffee": np.array([0, 1, 1, 0, 1]),
                "Sparkling Wine": np.array([0, 1, 1, 0, 1]),
                "White Wine": np.array([0, 1, 1, 0, 1]),
                "Red Wine": np.array([0, 1, 1, 0, 1]),
                "Domestic Beer": np.array([0, 1, 1, 0, 1]),

            },
            "HOT BEVERAGES": {
                "Gourmet Brewed Coffee": np.array([1, 1, 0, 0, 1]),
                "Espresso": np.array([0, 0, 1, 1, 0]),
                "Cappuccino": np.array([0, 0, 1, 1, 0]),
                "Caffé Latte": np.array([0, 0, 1, 1, 0]),
                "Matcha Latte": np.array([0, 0, 1, 1, 0]),
                "Caffé Americano": np.array([0, 0, 1, 1, 0]),
                "Caffé Mocha": np.array([0, 0, 1, 1, 0]),
                "Maple Latte": np.array([0, 0, 1, 1, 0]),
                "Hot Chocolate": np.array([0, 0, 1, 1, 0]),
                "Espresso": np.array([0, 0, 1, 1, 0]),
                
            },
             "COLD BEVERAGES": {
                "FRESH Orange Juice": np.array([1, 1, 0, 0, 1]),
                "Fruit Smoothies": np.array([0, 0, 1, 1, 0]),
                "Juice": np.array([0, 0, 1, 1, 0]),
                "Iced Latte": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
             "DRINKS": {
                "Kombucha Mimosa": np.array([1, 1, 0, 0, 1]),
                "Vegan Caesar": np.array([0, 0, 1, 1, 0]),
                "Vegan Latte": np.array([0, 0, 1, 1, 0]),
                "Vegan Hot Chocolate": np.array([0, 0, 1, 1, 0]),
                "Iced Vegan Latte": np.array([0, 0, 1, 1, 0]),
                "Kombucha": np.array([0, 0, 1, 1, 0]),
                "Milk": np.array([0, 0, 1, 1, 0]),
                "Strongbow Cider": np.array([0, 0, 1, 1, 0]),
            },
             "SALADS": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
             "VEGAN OMELETTE": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
             "VEGAN BENEDICT": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
             "VEGAN ALL DAY BREAKFAST": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
             "KIDS MENU": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "ALL DAY BREAKFAST": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "OMELETTES": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "WAFFLES": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "FROM THE GRILL": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "SANDWICHES & BURGERS": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "ADDITIONAL SIDE ORDERS": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            },
            "KIDS DRINK": {
                "Evviva Vegan House Salad": np.array([1, 1, 0, 0, 1]),
                "Vegan Kale Caesar Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Greek Salad": np.array([0, 0, 1, 1, 0]),
                "Vegan Chopped Salad": np.array([0, 0, 1, 1, 0]),
                "Milk or Chocolate Milk": np.array([0, 0, 1, 1, 0]),
                "Soft Drink": np.array([0, 0, 1, 1, 0]),
                "Evian Spring Water": np.array([0, 0, 1, 1, 0]),
                "Sparkling Water": np.array([0, 0, 1, 1, 0]),
            }
        }
        def recommend(self, category, userpreferences):
            # Compute similarities for a specific category
            similarities = {}
            for dish_name, dish_vector in self.dishes[category].items():
                similarity = cosine_similarity([userpreferences], [dish_vector])[0][0]
                similarities[dish_name] = similarity
            return similarities    

# Create an instance of the class outside the class definition
recommendation_engine = RecommendationEngine()


@app.route('/api/recommend/<string:category>', methods=['POST'])
def recommend(category):
     data = request.get_json()
     print("Data received:", data)
     userpreferences = np.array(data['preferences'])
     recommendations = recommendation_engine.recommend(category, userpreferences)

     userpreferences = data.get('preferences')
     if userpreferences is None:
         return jsonify({"message": "Missing preferences in request body"}), 400

     # Sort the recommendations based on the form data
     sorted_menu = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

     return jsonify(sorted_menu)


if __name__ == "__main__":
        app.run(debug=True)







 # Set up a route to handle POST requests
@app.route('/submit', methods=['POST'])
def get_form_data():
        form_data = request.get_json()

        # Process each category
        for category in form_data:
            form_data[category] = process_category(form_data[category])

        return jsonify(form_data)  # Send the processed data back as a response

if __name__ == '__main__':
    app.run(port=5000, debug=True)

@app.route('/submit', methods=['POST'])
def process_form():
            data = request.get_json()
            user_favorites = data['food'] + data['drink']

            # Call the compute_similarity function
            recommended_dishes = compute_similarity(user_favorites)

            return jsonify(recommended_dishes), 200

''')