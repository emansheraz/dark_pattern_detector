import pandas as pd

# Load your existing base file
try:
    df_existing = pd.read_csv("dark_patterns_dataset_clean.csv")
    print("Old dataset size:", df_existing.shape)
except FileNotFoundError:
    print("Could not find file. Starting fresh.")
    df_existing = pd.DataFrame(columns=['text', 'label'])

# ===== 1. HIDDEN_COST EXAMPLES =====
hidden_cost_examples = [
    "The final amount depends on your selections", "Your total may be recalculated at checkout",
    "Additional costs are the customer's responsibility", "Price does not cover all mandatory items",
    "Your payment total will be confirmed later", "There may be extra amounts due after purchase",
    "The listed price is only a part of what you owe", "You might be asked to pay more before delivery",
    "Some required items are not included in the base price", "Your order total is subject to adjustment",
    "The price you pay could be higher than shown", "Extra obligations may arise during checkout",
    "Your final bill is not yet determined", "Certain options will increase the total",
    "The displayed cost is incomplete", "Your actual expense may include unlisted items",
    "Price shown is a preliminary estimate", "You will see the full cost only at the last step",
    "Some parts of the total are not yet visible", "Your cart does not show all eventual charges",
    "The amount you pay may be revised upwards", "There are additional amounts you need to cover",
    "Your purchase total is not locked in yet", "The cost you see excludes certain necessities",
    "Your payment could include surprise elements", "The final amount is subject to change based on your choices",
    "Some features incur extra costs not displayed", "Your total is not guaranteed until payment",
    "The price you see is just the beginning", "Additional expenditure may be required later",
    "The final amount you owe may be higher than the initial quote.", "We will add any applicable taxes and fees before charging you.",
    "Your payment total is not final until we calculate all surcharges.", "Some mandatory costs are only shown after you click 'Checkout'.",
    "The advertised price does not reflect all required expenditures.", "Your cart currently shows only the subtotal - additional charges will be added.",
    "You agree to pay any extra fees determined at the time of delivery.", "The price you see may be increased by factors outside our control.",
    "Certain payment methods incur an additional percentage fee.", "Your total will include a variable fee based on your order weight.",
    "A handling fee is added to all returns, deducted from your refund.", "You will be charged a fee for canceling after the free period.",
    "A re-stocking charge applies to opened items.", "Your final bill includes a mandatory gratuity not shown earlier.",
    "The price displayed is for the base model; each additional feature costs extra.", "Your subscription price will increase after the promotional period.",
    "A one-time membership fee is added at checkout.", "Your total includes a 'convenience' charge for booking online.",
    "A processing fee is added to all ticket purchases.", "Your final price includes a facility maintenance surcharge.",
    "A 'smart delivery' fee is applied to expedited shipping.", "Your order includes a packaging fee for fragile items.",
    "A 'green' fee is added to offset carbon emissions.", "Your bill includes a regulatory recovery cost mandated by law.",
    "A 'consumer protection' fee is added to every transaction.", "Your total has a 'technology access' charge for using our platform.",
    "A 'data security' fee is applied to all online orders.", "Your invoice includes a 'paperless billing' fee even if you receive paper.",
    "A 'customer experience' fee is added for phone support.", "Your final charge includes a 'market adjustment' surcharge.",
    "A 'supply chain' fee is added due to current conditions.", "VAT not included in the displayed price",
    "Customs duties and import taxes are your responsibility", "A handling surcharge will be added for remote areas",
    "Your payment method may incur a cash advance fee", "Price shown is for the item only - packaging and delivery extra",
    "There is a minimum order value - below that, a small order fee applies", "Your total may include a fuel surcharge based on distance",
    "A restocking fee is deducted from refunds", "You will be charged a service fee for using our platform",
    "A convenience fee applies for credit card payments", "Your subscription renews automatically - cancel before the next billing cycle to avoid charges",
    "An activation fee is added to new accounts", "There is a one-time setup fee for this service",
    "Your order qualifies for an environmental recovery fee", "A regulatory compliance fee is added to all transactions",
    "A payment processing surcharge applies to this transaction", "Your total includes a technology fee for using our app",
    "A network usage charge is added for digital delivery", "A customer support fee is applied for phone orders",
    "A logistics coordination fee will be added to your invoice", "A warehouse storage fee may apply for delayed pickup",
    "A priority handling fee is optional but recommended", "A verification fee is required for identity confirmation",
    "A booking fee is added to each reservation", "A cleaning fee applies to this rental property",
    "A documentation fee is required for processing", "A title search fee is added to real estate transactions",
    "A credit report fee applies to loan applications", "A wire transfer fee is added for international payments",
    "A currency conversion fee applies to foreign transactions", "A cross-border processing fee is included in your total",
    "A brokerage fee may be added for imported goods", "A terminal handling charge is part of shipping costs"
]

# ===== 2. FAKE_URGENCY EXAMPLES =====
fake_urgency_examples = [
    "Only 2 items remaining in stock!", "Sale ends in 3 hours", "Hurry before it's gone", 
    "Last few items left", "Don't miss this deal", "Offer expires soon", "Act now while supplies last", 
    "Time-sensitive offer", "Order before midnight", "Exclusive limited-time sale", "Flash sale ending today", 
    "Quantities running low", "Reserved for early buyers", "Closing soon sale", "Price goes up in 10 minutes",
    "Only 1 room left at this price!", "High demand! 15 people are looking at this right now", 
    "Stock is running dangerously low", "Final clearance event ends tonight", "Grab yours before the timer hits zero",
    "Limited stock available now", "This deal is expiring in 5 minutes", "Almost sold out!", 
    "Items in your cart are selling fast", "Only 5 slots left for this webinar", "Don't wait! Prices increase tomorrow",
    "Last chance to claim your discount", "Flash sale: 50% off for the next 15 minutes", "Selling out quickly!",
    "Only a few units remaining", "Hurry! Demand is extremely high", "Offer valid for the next 60 seconds",
    "Get it before it's gone forever", "Supplies are strictly limited", "Final call for 40% off",
    "Prices return to normal in 2 hours", "Only 3 items left!", "Sale ends tonight!", "Limited time offer",
    "Hurry, stock running out", "Last chance to buy", "Sale expires in 2 hours", "Don't miss out!",
    "Offer valid for 24 hours only", "Hurry! Only 2 items left!", "Act fast! Limited inventory",
    "Deals like this don't last", "Strict limit of 2 per customer due to low stock", "Going fast!",
    "Unlock this price before it expires", "One-time offer expiring soon", "Immediate action required for discount",
    "Only 4 left in this color", "Demand is soaring, buy now", "Secure your order before we run out",
    "This offer is strictly time-limited", "Count down is on! Sale ends soon", "Available for today only",
    "Tomorrow this deal will be gone", "Last opportunity to purchase at this rate", "Stock moving fast",
    "Don't lose your spot, reserve now", "Only a handful remaining", "Liquidation sale ends in 1 hour",
    "Prices are about to rise", "Beat the clock! Shop now", "Final hours to save big",
    "Hurry, low stock alert!", "Item is currently in 45 shopping carts", "Limited production run, buy now",
    "Get yours before sell out", "Offer limited to the first 50 buyers", "Only 7 remaining",
    "Seize the deal before it terminates", "Midnight deadline approaching fast", "Don't let this slip away",
    "Inventory is running thin", "Last day to grab this exclusive item", "Limited edition selling fast",
    "Rooms are booking up quickly, lock in your rate", "Only 1 available at this special price",
    "Price drops like this are rare and temporary", "Hurry up, warehouse is almost empty",
    "Final batch available now", "Don't delay your purchase", "This promo code expires in 30 minutes",
    "Stock depletion warning!", "Only 9 slots remaining for early bird pricing", "Act now or miss out",
    "Sale is wrapping up!", "We are down to our final units", "Instant savings disappear at midnight",
    "Limited supply remaining", "Get it today or pay full price later", "Hurry, timer is ticking!"
]

# ===== 3. DISGUISED_AD EXAMPLES =====
disguised_ad_examples = [
    "Recommended just for you", "Customers also viewed this", "Trending now", "Featured product", 
    "Sponsored recommendation", "Popular items section", "You might also like", "Based on your browsing history", 
    "Top rated products", "Best seller in this category", "People are buying this right now", "Influencer pick of the week", 
    "Editor's choice", "Staff favorites", "Customer favorites", "Sponsored content", "Promoted products", 
    "Featured items", "Customers like you purchased", "Best sellers you might like", "Customers also bought",
    "You may also be interested in", "Handpicked for your style", "Products matching your interests",
    "Frequently bought together with this", "More items to explore", "Discover similar products",
    "Paid advertisement recommendation", "Our top picks for you", "Items you might love",
    "Trending in your area", "What others are looking at", "Selected just for you", "Top recommendations",
    "Most popular items today", "People who bought this also liked", "Highly rated by users like you",
    "Sponsored links from our partners", "Promoted item of the day", "Suggested products for your cart",
    "Items matching your recent searches", "Check out these related items", "People also shopped for",
    "Bestselling alternatives", "Products you shouldn't miss", "We think you'll love these",
    "Sponsored post", "Promoted search result", "Advertised merchandise", "Partner content",
    "Deals chosen for you", "Explore these curated options", "Our community loves these items",
    "Top choice for your next project", "Items trending globally", "Hot items right now",
    "Sponsored item listing", "Paid promotion", "Affiliate recommendation", "Featured collection",
    "Items selected based on your profile", "Similar items top rated", "People also viewed",
    "Highly requested products", "Products on our radar", "Buyer recommendations",
    "Our partners suggest", "Paid feature product", "Sponsored shopping result", "Brand spotlight item",
    "Most viewed items this week", "Top trending items in shopping", "Items frequently added to wishlists",
    "Recommended by experts", "Staff product pick", "Sponsored placement", "Promoted by seller",
    "Products popular with your demographic", "Items with highest repeat purchases", "Discovered for you",
    "Paid placement ad", "Sponsored product ad", "Promoted suggestion", "Recommended from our catalog",
    "Top choice among verified buyers", "Items frequently paired with your selection", "Hot products list",
    "Trending items from top brands", "Sponsored listing for you", "Promoted deal of the week",
    "Curated suggestions for your profile", "Items matching your shopping behavior", "Top selling items this month",
    "Sponsored highlight", "Promoted showcase", "Advertised selection"
]

# ===== 4. NORMAL EXAMPLES =====
normal_examples = [
    "Price includes all taxes", "Free shipping on all orders", "No additional fees apply", 
    "Total price is final", "All costs are included in the price", "You will not be charged extra", 
    "Final price is displayed upfront", "No hidden charges", "The amount shown is what you pay", 
    "Everything is included in the listed price", "Transparent pricing with no extra fees", 
    "No additional costs will be added", "The price is fixed and final", "All fees are already included", 
    "No surprises at checkout", "30-day money back guarantee", "Free returns on all items", 
    "Contact our support team anytime", "Shipping takes 5-7 business days", "Quality checked before shipping", 
    "Authentic product guaranteed", "Standard delivery available", "Express shipping option at checkout", 
    "Track your order here", "Customer service available 24/7", "We offer free returns within 30 days", 
    "Free shipping on orders over $50", "Made with premium materials", "Shop now for great deals", 
    "Product measurements: 10x12 inches", "Machine washable, tumble dry low", "100% organic cotton fibers", 
    "Designed and manufactured in the USA", "Includes 1-year manufacturer warranty", "Read our full privacy policy here", 
    "Secure checkout with SSL encryption", "Our store locations and operating hours", "Frequently asked questions about our service",
    "Download the official user manual PDF", "Package contents: 1 device, 2 cables, 1 charger",
    "Ingredients: Water, Glycerin, Citric Acid", "Compatible with iOS and Android devices",
    "Cancel your membership at any time from settings", "Track your package using the tracking number provided",
    "Sign up for our monthly newsletter updates", "Standard retail price excludes local sales tax",
    "Orders are processed within 24 hours of payment", "We accept all major credit cards and PayPal",
    "Review our terms and conditions of service", "This item meets all national safety standards",
    "Customer reviews are verified by an independent third party", "Product color options: Blue, Red, Black",
    "Eco-friendly packaging materials used", "No assembly required for this item",
    "Return policy details can be found on our help page", "We protect your data according to GDPR laws",
    "View your complete purchase history in your account dashboard", "Shipping calculated based on your destination postal code",
    "Product dimensions weight approximately 2.5 pounds", "Energy star certified appliance",
    "Our support team typically replies within 12 hours", "Subscribe to get occasional product announcements",
    "Thank you for supporting our family-owned business", "Detailed sizing chart available on the product page",
    "This software is updated regularly for security patches", "Your feedback helps us improve our services",
    "Bulk pricing discounts are available upon request", "Gift wrapping options are available for an extra fee",
    "Contact information for our corporate headquarters", "Read our commitment to corporate sustainability",
    "All prices are listed in USD by default", "Our customer satisfaction rating is currently 4.8 out of 5",
    "This item is currently backordered but shipping resumes next week", "How to properly care for your new leather boots",
    "Frequently asked customer questions answered below", "We do not sell your personal data to third parties",
    "Check out our updated blog for industry tips", "User accounts can be permanently deleted upon request",
    "Standard shipping rates apply to international orders", "Our materials are ethically sourced from sustainable farms",
    "This tutorial guide helps you setup your device step by step", "A receipt has been sent to your registered email address",
    "Warranty registration form must be filled out within 30 days", "All items are subject to a final quality control inspection",
    "Please read the safety guidelines before operating the machinery", "Customer service hotline phone number available on contact page"
]

new_data = []

# Append all 4 classes
for text in hidden_cost_examples:
    new_data.append({"text": text, "label": "hidden_cost"})

for text in fake_urgency_examples:
    new_data.append({"text": text, "label": "fake_urgency"})

for text in disguised_ad_examples:
    new_data.append({"text": text, "label": "disguised_ad"})

for text in normal_examples:
    new_data.append({"text": text, "label": "normal"})

# Convert to DataFrame
df_new = pd.DataFrame(new_data)
print("New data size:", df_new.shape)

# Merge with existing
df_updated = pd.concat([df_existing, df_new], ignore_index=True)

# Drop exact duplicates to keep data clean
df_updated = df_updated.drop_duplicates(subset=["text"])

# ==========================================
# THE FIX: AUTOMATICALLY BALANCE THE CLASSES
# ==========================================
# 1. Find the size of the smallest class
min_class_size = df_updated['label'].value_counts().min()
print(f"Smallest class size: {min_class_size}")

# 2. Shrink all larger classes down to match the smallest class
df_balanced = df_updated.groupby('label').sample(n=min_class_size, random_state=42)

# 3. Save the perfectly balanced dataset
df_balanced.to_csv("dark_patterns_dataset_clean.csv", index=False)

print("\n--- AUGMENTATION COMPLETE ---")
print("✅ Dataset updated and perfectly balanced!")
print("Final dataset size:", df_balanced.shape)
print(f"\nFinal distribution:")
for label, count in sorted(df_balanced['label'].value_counts().items()):
    print(f"  {label:15s}: {count} examples")
print(f"\n📁 Dataset saved to: dark_patterns_dataset_clean.csv")
print(f"✅ This is the file to use for training!")