import random

def get_recipe(food_item):
    recipes = {
        "burger": "Recipe for Burger:\n1. Ingredients: Burger buns, patty, lettuce, tomato, cheese, onion, pickles, ketchup, mustard\n2. Instructions: Grill patty, assemble burger with ingredients, serve hot.",
        "butter naan": "Recipe for Butter Naan:\n1. Ingredients: Flour, yogurt, baking soda, salt, sugar, butter\n2. Instructions: Knead dough, roll out, cook on skillet, brush with butter, serve warm.",
        "chai": "Recipe for Chai:\n1. Ingredients: Water, tea leaves, milk, sugar, ginger, cardamom\n2. Instructions: Boil water with tea leaves and spices, add milk and sugar, simmer, strain, and serve.",
        "chapati": "Recipe for Chapati:\n1. Ingredients: Whole wheat flour, water, salt\n2. Instructions: Knead dough, roll into discs, cook on skillet, serve.",
        "chole bhature": "Recipe for Chole Bhature:\n1. Ingredients: Chickpeas, flour, yogurt, spices, oil\n2. Instructions: Cook chickpeas with spices, make dough, deep fry bhature, serve.",
        "dal makhani": "Recipe for Dal Makhani:\n1. Ingredients: Black lentils, kidney beans, cream, butter, tomatoes, spices\n2. Instructions: Cook lentils and beans, add tomatoes and spices, simmer with cream and butter, serve.",
        "dhokla": "Recipe for Dhokla:\n1. Ingredients: Gram flour, yogurt, turmeric, Eno, mustard seeds, curry leaves\n2. Instructions: Mix ingredients, steam batter, temper with mustard seeds and curry leaves, serve.",
        "fried rice": "Recipe for Fried Rice:\n1. Ingredients: Cooked rice, vegetables, soy sauce, eggs, oil\n2. Instructions: Stir-fry vegetables, add rice and soy sauce, cook eggs separately, combine and serve.",
        "idli": "Recipe for Idli:\n1. Ingredients: Rice, urad dal, fenugreek seeds, salt\n2. Instructions: Soak and grind ingredients, ferment batter, steam in molds, serve.",
        "jalebi": "Recipe for Jalebi:\n1. Ingredients: Flour, yogurt, saffron, sugar, ghee\n2. Instructions: Make batter, ferment, pipe into hot oil, soak in sugar syrup, serve.",
        "kaathi rolls": "Recipe for Kaathi Rolls:\n1. Ingredients: Parathas, chicken or paneer, spices, chutney, onions\n2. Instructions: Cook filling, assemble roll with parathas and chutney, serve.",
        "kadai paneer": "Recipe for Kadai Paneer:\n1. Ingredients: Paneer, bell peppers, onions, tomatoes, spices\n2. Instructions: Cook vegetables and paneer with spices in a kadai, serve.",
        "kulfi": "Recipe for Kulfi:\n1. Ingredients: Milk, sugar, cardamom, pistachios\n2. Instructions: Simmer milk and sugar until thick, add cardamom and nuts, freeze, serve.",
        "masala dosa": "Recipe for Masala Dosa:\n1. Ingredients: Rice, urad dal, potato, spices\n2. Instructions: Make batter, cook dosa, fill with spiced potato mixture, serve with chutney.",
        "momos": "Recipe for Momos:\n1. Ingredients: Flour, cabbage, carrot, soy sauce, ginger, garlic\n2. Instructions: Make dough, prepare filling, steam momos, serve with sauce.",
        "paani puri": "Recipe for Paani Puri:\n1. Ingredients: Semolina, flour, tamarind, spices, potatoes, chickpeas\n2. Instructions: Make puris, prepare fillings and tamarind water, assemble and serve.",
        "pakode": "Recipe for Pakode:\n1. Ingredients: Gram flour, onions, potatoes, spices\n2. Instructions: Make batter, dip vegetables, deep fry, serve with chutney.",
        "pav bhaji": "Recipe for Pav Bhaji:\n1. Ingredients: Mixed vegetables, pav bread, butter, spices\n2. Instructions: Cook vegetables with spices, mash, serve with buttered pav.",
        "pizza": "Recipe for Pizza:\n1. Ingredients: Pizza dough, tomato sauce, cheese, toppings\n2. Instructions: Prepare dough, add sauce and toppings, bake in oven, serve.",
        "samosa": "Recipe for Samosa:\n1. Ingredients: Flour, potatoes, peas, spices, oil\n2. Instructions: Make dough, prepare filling, shape and deep fry samosas, serve."
    }

    return recipes.get(food_item.lower(), "No recipe found for this food item. Try a different search.")


