import joblib


class RecommendationSystem():
    """ This class will contain the code to connect the recommendaion system with other sentiment analysis model.
    """
    
    def __init__(self):
        self.df_reviews = joblib.load(r'models/processed_reviews_df.pkl')
        self.sentiment_model = joblib.load(r'models/sentiment_model.pkl')
        self.user_ratings = joblib.load(r'models/user_ratings.pkl')
        self.vectorizer = joblib.load(r'models/vectorizer.pkl')
        self.user_dict = joblib.load(r'models/user_dictionary.pkl')
        self.product_dict = joblib.load(r'models/product_dictionary.pkl')
    
    
    def recommend_products(self, username):
        """This function takes username as input and gives top 5 recommend items.

        Args:
            username (text): Username of the user in dataset

        Returns:
            tuple: Product list, is username correct or not
        """
        # Check whether username is proper or not
        if username not in self.user_dict:
            return ("", False)

        # get the user_id for the user name associated
        # Recommending the Top 20 products to the user.
        product_id_list = self.user_ratings.loc[self.user_dict[username]].sort_values(ascending=False)[0:20]

        # Convert product id list to product names
        product_names = []
        for i in product_id_list.index:
            product_names.append(self.product_dict[i])
            
        # Pass the above list to sentiment model and find the 5 top most products
        product_positive_scores = []
        for product in product_names:
            # Find all the relevant reviews for the specific product
            product_reviews = self.df_reviews[self.df_reviews.name == product].processed_review
            
            #Convert the reviews to numerical features
            X_values = self.vectorizer.transform(product_reviews)
            
            # Find the average positivity rate for the product
            positive_sum = 0
            probs = self.sentiment_model.predict_proba(X_values)
            for x in probs:
                positive_sum += x[1]
            product_positive_scores.append(positive_sum/len(probs))

        product_positive_zip = zip(product_names, product_positive_scores)

        # Sort based on positivity rate
        product_positive_zip_sorted = sorted(product_positive_zip, key=lambda x:x[1], reverse=True)
        final_product_list, _ =zip(*product_positive_zip_sorted)
        return (final_product_list[:5],True)