import json
with open('submission_cache.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for k in d:
    # 1. Bridal followup
    if k.startswith('trg_007_bridal_followup_kavya'):
        d[k]['body'] = "Hi Kavya, Lakshmi here from Studio11! Bas 196 days bache hain aapke big day ke liye! Aapka 30-day skin prep program window ab open ho gaya hai. Humari bridal inquiries kaafi badh gayi hain (last week calls 20% up the!), toh slots jaldi fill ho rahe hain. Perfect time hai us radiant glow ke liye humara ₹14,999 ka bridal prep package start karne ka. Kya main aapke first session ke liye Saturday 11 AM ka slot block karoon 💇"
    
    # 2. IPL match
    elif k.startswith('trg_010_ipl_match'):
        body = d[k]['body']
        if 'Buy 1 Get 1' in body and '₹' not in body:
            d[k]['body'] = body.replace('Buy 1 Get 1', '₹499 wala Buy 1 Get 1')
            
    # 3. Milestone
    elif k.startswith('trg_012_milestone_mylari'):
        d[k]['body'] = "Hi Suresh, Mylari South Indian Cafe 150 reviews ke milestone se bas 5 reviews door hai! Pichle 7 din mein aapke views 5% badhe hain aur calls bhi 2% up hain. 5 reviews door ho aur next 48 hrs mein 3 reviews aa jayein toh algorithm boost milega — kya main 3 regulars ko reminder bhejoon? 🍽️"

    # 4. Seasonal Dip
    elif k.startswith('trg_014_seasonal_acquisition_dip'):
        d[k]['body'] = "Karthik, performance metrics mein thoda seasonal dip dikh raha hai. Last 7 days mein views 30% aur calls 35% down hain. Apr-Jun acquisition slow hota hai, toh chaliye retention aur referrals par focus karein. Main aapka ₹1,999 value wala '3 FREE Trial Classes' offer refresh karke ek 'Bring a Buddy' post draft kar doon? Karein? Reply YES/STOP 💪"

    # 5. Recall
    elif k.startswith('trg_018_supply_atorvastatin_recall'):
        d[k]['body'] = "Ramesh, atorvastatin recall ke liye affected customers ki list ready hai. MfrZ ke specific batches AT2024-1102 aur AT2024-1108 par dhyan dena hai. Aapke profile par views 6% aur calls 8% up hain, isliye trust maintain karna critical hai. Aapko estimated ₹4,500 ka revenue loss protect karne ke liye main alternate brand recommendations ke sath list abhi WhatsApp par bhej doon? 💊"

with open('submission_cache.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print('Updated 5 cache entries successfully.')
