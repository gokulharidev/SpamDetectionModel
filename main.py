from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# This is a simple dataset. In a real-world scenario, you'd likely use a larger dataset.
email = [
    'You won a free ticket to the USA this summer.',
    'Congratulations! You have been selected for a cash prize.',
    'Click here to claim your reward.',
    'You have won $1000 cash!',
    'Exclusive offer just for you.',
    'Don\'t miss this chance!',
    'Your email has been selected.',
    'Earn money without leaving your home.',
    'Increase your income.',
    'Get rid of debt.',
    'You\'re a winner!',
    'This is your lucky day.',
    'Get your prize now.',
    'Free membership.',
    'Unlimited access.',
    'Important information regarding your account.',
    'Your account might be compromised.',
    'Login to your account immediately.',
    'Update your payment information.',
    'Your credit card might be suspended.',
    'Immediate action required.',
    'Your order is ready.',
    'Get started with your free trial.',
    'This is a limited time offer.',
    'Satisfaction guaranteed.',
    'Money back guarantee.',
    'We miss you.',
    'See what you missed.',
    'Start earning points today.',
    'You earned a reward.',
    'Your opinion matters.',
    'Take our quick survey.',
    'You\'ve received a new message.',
    'Your subscription is ending.',
    'Renew your subscription.'
]

labels = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert the text data into numerical data for the model
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(email)

# Split the data into training and test sets
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(features, labels)

#real email

import os
from email import policy
from email.parser import BytesParser

# Directory containing your EML files
directory=r'C:\Users\gokul\OneDrive\Desktop\trainingspace\machine learning\supervised learning\classification\spamdetection.py\emails'


# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if the file is an EML file
    if filename.endswith('.eml'):
        # Open the file
        with open(os.path.join(directory, filename), 'rb') as f:
            # Parse the EML file
            msg = BytesParser(policy=policy.default).parse(f)
            # Extract the subject and body
            subject = msg['subject']
            body = msg.get_body(preferencelist=('plain')).get_content()
            # Concatenate the subject and body
            email_text = subject + ' ' + body
            # Transform the email data
            feature_test = vectorizer.transform([email_text])
            # Make a prediction
            prediction = clf.predict(feature_test)
            # Print the prediction
            if prediction == 1:
                print('{} is Spam'.format(filename))
            else:
                print('{} is not Spam'.format(filename))
            
        