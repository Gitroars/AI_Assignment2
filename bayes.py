import pandas as pd
import requests

url = 'https://raw.githubusercontent.com/Gitroars/AI_Assignment2/main/content.csv'
df = pd.read_csv(url)
print(df)

#region Probability of Days and Purchase
# Extract unique days
unique_days = df['Day'].unique()
purchase_dict = {}
# Iterate through unique days and create the dictionary
for day in unique_days:
    day_data = df[df['Day'] == day]
    purchase_counts = day_data['Purchase'].value_counts().to_dict()
    purchase_dict[day] = purchase_counts

print(purchase_dict)

# Calculate the prior probability of each day type
prior_probs = df['Day'].value_counts(normalize=True).to_dict()

# Calculate the conditional probability of each purchase type given the day type
conditional_probs = {}
for day in prior_probs.keys():
    day_data = df[df['Day'] == day]
    conditional_probs[day] = day_data['Purchase'].value_counts(normalize=True).to_dict()

# Calculate the overall probability of each purchase type
overall_probs = df['Purchase'].value_counts(normalize=True).to_dict()

# Calculate the posterior probability for every pair of day type and Yes/No purchase
day_posterior_probs = {}
for day in prior_probs.keys():
    day_posterior_probs[day] = {}
    for purchase in overall_probs.keys():
        posterior_prob = (prior_probs[day] * conditional_probs[day].get(purchase, 0)) / overall_probs[purchase]
        day_posterior_probs[day][purchase] = posterior_prob

# Print the posterior probabilities
for day in day_posterior_probs.keys():
    for purchase, probability in day_posterior_probs[day].items():
        print(f'P({day}|{purchase}): {probability}')
#endregion
#region Probability of Discount and Purchase
# Calculate the prior probability of each combination of Discount and Purchase
prior_probs_discount_purchase = df.groupby(['Discount', 'Purchase']).size() / len(df)

# Calculate the overall probability of each Purchase type
overall_probs_purchase = df['Purchase'].value_counts(normalize=True).to_dict()

# Calculate the posterior probabilities for each combination of Discount and Purchase
discount_posterior_probs = {}
for (discount, purchase), prior_prob in prior_probs_discount_purchase.items():
    posterior_prob = (prior_prob * overall_probs_purchase.get(purchase, 0)) / prior_probs_discount_purchase[discount, purchase]
    discount_posterior_probs[(discount, purchase)] = posterior_prob

# Print the posterior probabilities
for (discount, purchase), probability in discount_posterior_probs.items():
    print(f'P({purchase}|{discount}): {probability}')

#endregion

#region Probability of Free Delivery and Purchase
# Calculate the prior probability of each "Free Delivery" type (Yes/No)
prior_probs = df['Free Delivery'].value_counts(normalize=True).to_dict()

# Calculate the conditional probability of each "Purchase" type (Yes/No) given the "Free Delivery" type
conditional_probs = {}
for delivery in prior_probs.keys():
    delivery_data = df[df['Free Delivery'] == delivery]
    conditional_probs[delivery] = delivery_data['Purchase'].value_counts(normalize=True).to_dict()

# Calculate the overall probability of each "Purchase" type (Yes/No)
overall_probs = df['Purchase'].value_counts(normalize=True).to_dict()

# Calculate the posterior probability for every pair of "Free Delivery" type and "Purchase" type
delivery_posterior_probs = {}
for delivery in prior_probs.keys():
    delivery_posterior_probs[delivery] = {}
    for purchase in overall_probs.keys():
        posterior_prob = (prior_probs[delivery] * conditional_probs[delivery].get(purchase, 0)) / overall_probs[purchase]
        delivery_posterior_probs[delivery][purchase] = posterior_prob

# Print the posterior probabilities
for delivery in delivery_posterior_probs.keys():
    for purchase, probability in delivery_posterior_probs[delivery].items():
        print(f'P({purchase}|{delivery}): {probability}')

#endregion


input_day = 'Weekday'
input_discount = 'No'
input_delivery = 'Yes'

posterior_prob_day_yes = day_posterior_probs[input_day]['Yes']
posterior_prob_discount_yes = discount_posterior_probs[(input_discount,'Yes')]
posterior_prob_delivery_yes = discount_posterior_probs[(input_delivery,'Yes')]
answer_yes = posterior_prob_day_yes * posterior_prob_discount_yes * posterior_prob_delivery_yes
print("P(Yes)",answer_yes)

# Calculate the posterior probability of "No" purchase given the input day, discount, and delivery
posterior_prob_day_no = day_posterior_probs[input_day]['No']
posterior_prob_discount_no = discount_posterior_probs[(input_discount,'No')]
posterior_prob_delivery_no = delivery_posterior_probs[input_delivery]['No']
answer_no = posterior_prob_day_no * posterior_prob_discount_no * posterior_prob_delivery_no
print("P(No)",answer_no)

# Calculate the total probability of all outcomes
total_prob = answer_yes + answer_no

# Normalize the posterior probabilities
posterior_prob_yes = answer_yes / total_prob
posterior_prob_no = answer_no / total_prob

print("Purchase:",posterior_prob_yes)
print("Not Purchase:",posterior_prob_no)