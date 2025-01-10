def generate_response(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Nutrition-related questions
    if "nutrition" in user_input:
        return (
            "Eating healthy food is essential because it provides your body with the nutrients needed "
            "for growth, repair, and overall well-being. A balanced diet helps prevent chronic diseases "
            "and boosts energy levels."
        )

    # Food origin-related questions
    elif "origin" in user_input:
        food_origins = {
            "burger": "The burger originated in Germany but gained popularity in the United States.",
            "butter naan": "Butter naan has its origins in Indian cuisine, specifically as a North Indian delicacy.",
            "chai": "Chai, or tea, originated in India and is an integral part of Indian culture.",
            "chapati": "Chapati is a staple in Indian cuisine, particularly in North and Western India.",
            "chole bhature": "Chole Bhature is a popular North Indian dish from Punjab.",
            "dal makhani": "Dal Makhani originated in Punjab, India, and is known for its creamy texture.",
            "dhokla": "Dhokla is a fermented dish that originates from Gujarat, India.",
            "fried_rice": "Fried rice traces its origins to China but is now popular worldwide with regional variations.",
            "idli": "Idli is a South Indian delicacy with origins in Tamil Nadu and Karnataka.",
            "jalebi": "Jalebi is thought to have Persian origins but is a popular sweet in India.",
            "kaathi rolls": "Kathi rolls originated in Kolkata, India, as a street food dish.",
            "kadai paneer": "Kadai Paneer is a North Indian curry dish made with cottage cheese and spices.",
            "kulfi": "Kulfi is a traditional Indian frozen dessert that originated during the Mughal era.",
            "masala dosa": "Masala Dosa is a South Indian dish from Karnataka, popular throughout India.",
            "momos": "Momos are a Tibetan delicacy that has become popular in North and Northeast India.",
            "paani puri": "Paani Puri, also known as Golgappa, originated in India and is a popular street food.",
            "pakode": "Pakode, or fritters, are a common Indian snack with roots in traditional Indian kitchens.",
            "pav bhaji": "Pav Bhaji originated in Mumbai as a quick and hearty meal for mill workers.",
            "pizza": "Pizza originated in Italy and is now a globally beloved dish.",
            "samosa": "Samosa has Persian origins but was introduced to India during the Delhi Sultanate."
        }
        # Extract the food item from the user input
        for food, origin in food_origins.items():
            if food in user_input:
                return f"The origin of {food} is: {origin}"

        return "Could you specify which food item you'd like to know the origin of?"

    # Basic food-related questions
    elif "healthy food" in user_input:
        return (
            "Eating healthy food is essential to maintain a balanced diet, support mental and physical health, "
            "and prevent chronic diseases like obesity, diabetes, and heart conditions."
        )
    elif "how to make" in user_input:
        return "Let me know the dish you're interested in, and I'll suggest a recipe!"
    elif "what is" in user_input:
        return (
            "Could you clarify your question? Are you asking about a specific food item, its recipe, or nutritional value?"
        )

    # Dataset-specific questions
    elif "dataset" in user_input:
        return (
            "Our dataset includes images of various Indian and international foods like burgers, butter naan, chai, "
            "chapati, chole bhature, dal makhani, and more. Let me know if you'd like details about any of these dishes!"
        )

    # Recipe suggestions
    elif "recipe" in user_input:
        return (
            "I can suggest recipes for items like Butter Naan, Dal Makhani, Masala Dosa, and more. What would you like to cook?"
        )

    # General fallback
    else:
        return (
            "I'm here to answer your food-related questions! Could you ask about nutrition, recipes, origins, or a specific dish?"
        )
