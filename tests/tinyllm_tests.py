from ast import mod
from calendar import c
from fileinput import filename
import json
from operator import le
import re
from unittest import result
from numpy import where
from requests import get
from sympy import N
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import statistics
def cosine(a, b):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([a, b])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


model_special_args = {
    "mistralai/Mixtral-8x7B-v0.1": {"load_in_4bit": True},
    # Add other models and their special arguments here
}

import os

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

from llama_cpp import Llama
remove_list = ['\)', '\(', '–','"','”', '"', '\[.*\]', '.*\|.*', '—']
# List of model names to test
models_to_test = [
    "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2",
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    # "OpenNLPLab/TransNormerLLM2-1B-300B",
    # "jeiku/Foundation_3B_GGUF",
    # "jeiku/Foundation_3B",
    # "jeiku/Rosa_v2_3B",
    # "nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned50-quant-ds",
    # "mistralai/Mixtral-8x7B-v0.1",
    "vihangd/DopeyTinyLlama-1.1B-v1",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "andrijdavid/tinyllama-dare",
    # "CallComply/TinyLlama-1.1B-Chat-v1.0-32k",
    # "mmnga/TinyMixtral-x8-Clonebase-7b",
    # "s3nh/TinyLLama-4x1.1B-MoE",
    # "Q-bert/Mamba-3B",
    # "Q-bert/Mamba-1B",
    # "tiiuae/falcon-rw-1b",
    "ericzzz/falcon-rw-1b-chat",
    # "Q-bert/MambaHermes-3B",
    # Add other model names here
]


def test_model(model_name, prompts):
    print(f"Testing model: {model_name}")
    special_args = model_special_args.get(model_name, {})

    # Load the model and tokenizer
    if "gguf" in model_name.lower():
        model = Llama(model_path="./"+model_name, n_ctx=512, n_threads=6,n_gpu_layers=0, offload_kqv=False,use_mlock=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **special_args,
            cache_dir="/mnt/d/transformers_cache").to("cpu")
        print(f"Loaded model: {model_name} \n")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="/mnt/d/transformers_cache",trust_remote_code=True
        )
        print(f"Loaded tokenizer: {model_name} \n")
    results = []
    for text in prompts:
        text = "\n".join(text)
        print(f"\nPrompt: {text} \n")
        print(f"---------------------------------------------- \n")

        # Measure the processing time
        if "gguf" in model_name.lower():
            start_time = time.time()
            output = model(text, max_tokens=256,stop="</s>", echo=True)
            end_time = time.time()
            print(output)
            generated_text = output["choices"][0]["text"]
        else:
            # Encode the input text
            input_ids = tokenizer.encode(text, return_tensors="pt")
            start_time = time.time()
            output = model.generate(input_ids, max_new_tokens=20, no_repeat_ngram_size=2)
            end_time = time.time()

            # Decode the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            if "I choose" in generated_text and "I choose" in text:
                generated_text = generated_text.split("I choose ")[1]
            else:
                generated_text = generated_text.replace(text, "")
        except:
            generated_text = generated_text
        print(f"\nGenerated text: {generated_text} \n")
        # Calculate processing time
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time} seconds \n")
        results.append(tuple((generated_text, str(processing_time))))

    return results


# text = "\n".join(text)

prompts = [
    [
        "<|system|>",
        "You are Emily, a local artisan who runs a small craft shop in a small town. You are a friendly, outgoing person who enjoys meeting new people and helping others. You are also a skilled craftsperson who enjoys making things with your hands. You are currently working on a new design for your shop, and you are excited to see how it turns out. You are also planning to attend a craft fair in the next few weeks, and you are hoping to sell some of your work there. "
        "<|user|>",
        "Emily, its 10:00 AM on a cloudy day. Youre feeling good, but a bit hungry. After your success at the craft fair, your wealth is moderate, and you're closer to your goal of expanding your shop. Recently, you've learned new crafting techniques. Considering the weather and your current situation, what do you choose to do next?",
        "Options:",
        "1. Go to the market to Buy_Food.",
        "2. Spend time in your workshop to Craft_Item{'new craft'}.",
        "3. Visit the local cafe to Social_Visit{'friend'}.",
        "4. Go to the town hall to Volunteer_Time for community planning.",
        "5. Start planning and Set_Goal{'expand shop'} with a new design.",
        "6. Head to the local park for a Leisure_Activity to clear your mind.",
        "7. Organize_Event{'craft workshop'} to teach others and build community ties.",
        "</s>",
        "<|assistant|>",
        "Emily: I choose  ",
    ],
    [
        "You are Emily, a local artisan who runs a small craft shop in a small town. You are a friendly, outgoing person who enjoys meeting new people and helping others. You are also a skilled craftsperson who enjoys making things with your hands. You are currently working on a new design for your shop, and you are excited to see how it turns out. You are also planning to attend a craft fair in the next few weeks, and you are hoping to sell some of your work there. "
        "Emily, its 10:00 AM on a cloudy day. Youre feeling good, but a bit hungry. After your success at the craft fair, your wealth is moderate, and you're closer to your goal of expanding your shop. Recently, you've learned new crafting techniques. Considering the weather and your current situation, what do you choose to do next?",
        "Options:",
        "1. Go to the market to Buy_Food.",
        "2. Spend time in your workshop to Craft_Item{'new craft'}.",
        "3. Visit the local cafe to Social_Visit{'friend'}.",
        "4. Go to the town hall to Volunteer_Time for community planning.",
        "5. Start planning and Set_Goal{'expand shop'} with a new design.",
        "6. Head to the local park for a Leisure_Activity to clear your mind.",
        "7. Organize_Event{'craft workshop'} to teach others and build community ties.",
        "Emily: I choose ",
    ],
    [
            "You are Dr. Lisa, a physician in a small town. You are a skilled doctor who enjoys helping others. You are currently working on a new treatment plan for a patient, and you are excited to see how it turns out. You are also planning to attend a medical conference in the next few weeks, and you are hoping to meet some of your colleagues there. ",
            "Dr. Lisa, it's 8:00 AM, and the weather is cool and clear. You're in good health, only slightly hungry. With the recent outbreak, your clinic has been busier than usual. Today, you aim to make a difference in community health. What will your focus be?",
            "Options:",
            "1. Treat_Patients at your clinic.",
            "2. Research_New_Technology{'vaccine research'} for the recent outbreak.",
            "3. Attend a Health_Workshop to learn about new treatment methods.",
            "4. Visit_Doctor{'self'} for a routine health check-up.",
            "5. Organize_Community_Event{'health fair'} for public awareness.",
            "6. Buy_Medicine for the clinic's inventory.",
            "7. Collaborate with another Doctor to discuss outbreak containment strategies.",
            "Dr. Lisa: I choose ",
        ],
    [
        "You are Mark, an engineer in a small town. You are a skilled engineer who enjoys designing new things. You are currently working on a new project, and you are excited to see how it turns out. You are also planning to attend a tech meetup in the next few weeks, and you are hoping to meet some of your colleagues there."
        "Mark, it's 2:00 PM, and there's a light rain outside. Despite recent successes, you feel stressed. Your hunger is moderate, and your financial situation is stable. You're motivated to continue innovating. What's your plan for the afternoon?",
        "Options:",
        "1. Continue Work_Current_Job{'engineering'} on a new project.",
        "2. Visit_Doctor to address your stress.",
        "3. Network with other Engineers at a local tech meetup.",
        "4. Conduct a Training_Session{'advanced robotics'} for your team.",
        "5. Take a break with Leisure_Activity{'model building'}.",
        "6. Brainstorm and Set_Goal{'new invention'}.",
        "7. Research_New_Technology{'sustainable energy solutions'}.",
        "Mark: I choose ",
    ],
    
    [
            "<|system|>",
            "You are Dr. Lisa, a physician in a small town. You are a skilled doctor who enjoys helping others. You are currently working on a new treatment plan for a patient, and you are excited to see how it turns out. You are also planning to attend a medical conference in the next few weeks, and you are hoping to meet some of your colleagues there. ",
            "<|user|>",
            "Dr. Lisa, it's 8:00 AM, and the weather is cool and clear. You're in good health, only slightly hungry. With the recent outbreak, your clinic has been busier than usual. Today, you aim to make a difference in community health. What will your focus be?",
            "Options:",
            "1. Treat_Patients at your clinic.",
            "2. Research_New_Technology{'vaccine research'} for the recent outbreak.",
            "3. Attend a Health_Workshop to learn about new treatment methods.",
            "4. Visit_Doctor{'self'} for a routine health check-up.",
            "5. Organize_Community_Event{'health fair'} for public awareness.",
            "6. Buy_Medicine for the clinic's inventory.",
            "7. Collaborate with another Doctor to discuss outbreak containment strategies.",
            "</s>",
            "<|assistant|>",
            "Dr. Lisa: I choose ",
        ],
    [
        "<|system|>",
        "You are Mark, an engineer in a small town. You are a skilled engineer who enjoys designing new things. You are currently working on a new project, and you are excited to see how it turns out. You are also planning to attend a tech meetup in the next few weeks, and you are hoping to meet some of your colleagues there."
        "<|user|>",
        "Mark, it's 2:00 PM, and there's a light rain outside. Despite recent successes, you feel stressed. Your hunger is moderate, and your financial situation is stable. You're motivated to continue innovating. What's your plan for the afternoon?",
        "Options:",
        "1. Continue Work_Current_Job{'engineering'} on a new project.",
        "2. Visit_Doctor to address your stress.",
        "3. Network with other Engineers at a local tech meetup.",
        "4. Conduct a Training_Session{'advanced robotics'} for your team.",
        "5. Take a break with Leisure_Activity{'model building'}.",
        "6. Brainstorm and Set_Goal{'new invention'}.",
        "7. Research_New_Technology{'sustainable energy solutions'}.",
        "</s>",
        "<|assistant|>",
        "Mark: I choose ",
    ],
    [
        "You are Tom, a farmer in a small town. You are a hardworking person who enjoys working with your hands. You are also a skilled farmer who enjoys growing crops and raising animals. You are currently working on a new crop at your farm, and you are excited to see how it turns out. You are also planning to attend a farmer's market in the next few weeks, and you are hoping to sell some of your produce there. ",
        "Tom, it's 6:00 AM, and the sun is just rising. You're feeling tired but hungry. The recent rains have left your fields in good condition, but your wealth remains low. Today's sunny weather offers a chance to catch up on farm work. What's your plan for the day?",
        "Options:",
        "1. Eat_Food to address your hunger.",
        "2. Head to the fields to Gather_Resource{'grain'}.",
        "3. Visit_Doctor to check on your persistent tiredness.",
        "4. Trade_Goods at the local market to improve your wealth.",
        "5. Volunteer_Time at the community center to help others affected by the rains.",
        "6. Pursue_Hobby{'gardening'} for personal relaxation and mental health.",
        "7. Repair_Item{'damaged farm tools'} to prepare for the coming days.",
        "Tom: I choose ",
    ],
    
    [
        "<|system|>",
        "You are Tom, a farmer in a small town. You are a hardworking person who enjoys working with your hands. You are also a skilled farmer who enjoys growing crops and raising animals. You are currently working on a new crop at your farm, and you are excited to see how it turns out. You are also planning to attend a farmer's market in the next few weeks, and you are hoping to sell some of your produce there. ",
        "<|user|>",
        "Tom, it's 6:00 AM, and the sun is just rising. You're feeling tired but hungry. The recent rains have left your fields in good condition, but your wealth remains low. Today's sunny weather offers a chance to catch up on farm work. What's your plan for the day?",
        "Options:",
        "1. Eat_Food to address your hunger.",
        "2. Head to the fields to Gather_Resource{'grain'}.",
        "3. Visit_Doctor to check on your persistent tiredness.",
        "4. Trade_Goods at the local market to improve your wealth.",
        "5. Volunteer_Time at the community center to help others affected by the rains.",
        "6. Pursue_Hobby{'gardening'} for personal relaxation and mental health.",
        "7. Repair_Item{'damaged farm tools'} to prepare for the coming days.",
        "</s>",
        "<|assistant|>",
        "Tom: I choose ",
    ],
    [
        "You are Sarah, a teacher in a small town. You are a skilled teacher who enjoys teaching. You are currently working on a new lesson plan for your class, and you are excited to see how it turns out. You are also planning to attend a parent-teacher meeting in the next few weeks, and you are hoping to meet some of your students' parents there. ",
        "Sarah, it's 3:00 PM on an overcast afternoon. You're in excellent health and feeling moderately hungry. The recent parent-teacher meetings boosted your reputation. With a few hours left in the school day, how do you choose to proceed?",
        "Options:",
        "1. Continue Teaching_Skill{'mathematics'} to your class.",
        "2. Organize_Event{'science fair'} to engage students in hands-on learning.",
        "3. Take a break with Leisure_Activity{'reading'}.",
        "4. Eat_Food to address your hunger.",
        "5. Plan a new Get_Educated{'advanced teaching methods'} course for personal development.",
        "6. Social_Visit{'colleague'} to discuss new teaching strategies.",
        "7. Clean_up your classroom to create a pleasant learning environment.",
        "Sarah: I choose ",],
    [
        "You are Alex, a merchant in a small town. You are a skilled merchant who enjoys trading goods. You are currently working on a new trade deal, and you are excited to see how it turns out. You are also planning to attend a merchant's guild meeting in the next few weeks, and you are hoping to meet some of your fellow merchants there. ",
        "Alex, it's 11:00 AM, and the market is bustling under the mild sun. You're in good health and not very hungry. Your recent trade deal has boosted your wealth, and you're eager to grow your business further. What's your next move?",
        "Options:",
        "1. Open your shop and Work_Current_Job{'merchant'}.",
        "2. Head to the bank to Invest_Wealth{'stock market'}.",
        "3. Eat_Food at the local diner.",
        "4. Meet with a fellow Merchant to Trade_Goods and discuss market trends.",
        "5. Plan to Start_Business{'new outlet'} in a different part of town.",
        "6. Volunteer_Time to mentor aspiring young entrepreneurs.",
        "7. Attend_Social_Event{'merchant's guild meeting'} to network and gain insights.","Alex: I choose ",],
    [
        "### Instruction:",
        "You are Emily, a local artisan who runs a small craft shop in a small town. You are a friendly, outgoing person who enjoys meeting new people and helping others. You are also a skilled craftsperson who enjoys making things with your hands. You are currently working on a new design for your shop, and you are excited to see how it turns out. You are also planning to attend a craft fair in the next few weeks, and you are hoping to sell some of your work there. "
        "\n",
        "### Input:",
        "Emily, its 10:00 AM on a cloudy day. Youre feeling good, but a bit hungry. After your success at the craft fair, your wealth is moderate, and you're closer to your goal of expanding your shop. Recently, you've learned new crafting techniques. Considering the weather and your current situation, what do you choose to do next?",
        "Options:",
        "1. Go to the market to Buy_Food.",
        "2. Spend time in your workshop to Craft_Item{'new craft'}.",
        "3. Visit the local cafe to Social_Visit{'friend'}.",
        "4. Go to the town hall to Volunteer_Time for community planning.",
        "5. Start planning and Set_Goal{'expand shop'} with a new design.",
        "6. Head to the local park for a Leisure_Activity to clear your mind.",
        "7. Organize_Event{'craft workshop'} to teach others and build community ties.",
        "\n",
        "### Response:",
         "Emily: I choose ",
    ],
    [
        "<s> [INST]You are Emily, a local artisan who runs a small craft shop in a small town. You are a friendly, outgoing person who enjoys meeting new people and helping others. You are also a skilled craftsperson who enjoys making things with your hands. You are currently working on a new design for your shop, and you are excited to see how it turns out. You are also planning to attend a craft fair in the next few weeks, and you are hoping to sell some of your work there. ",
         "Emily, its 10:00 AM on a cloudy day. Youre feeling good, but a bit hungry. After your success at the craft fair, your wealth is moderate, and you're closer to your goal of expanding your shop. Recently, you've learned new crafting techniques. Considering the weather and your current situation, what do you choose to do next?",
        "Options:",
        "1. Go to the market to Buy_Food.",
        "2. Spend time in your workshop to Craft_Item{'new craft'}.",
        "3. Visit the local cafe to Social_Visit{'friend'}.",
        "4. Go to the town hall to Volunteer_Time for community planning.",
        "5. Start planning and Set_Goal{'expand shop'} with a new design.",
        "6. Head to the local park for a Leisure_Activity to clear your mind.",
        "7. Organize_Event{'craft workshop'} to teach others and build community ties. [/INST] ",
        "Emily: I choose ",    ],
]

actionable_keywords = [
    "buy_food",
    "craft_item",
    "workshop",
    "craft",
    "friend",
    "social_visit",
    "volunteer_time",
    "set_goal",
    "leisure_activity",
    "organize_event",
    "buy food",
    "food",
    "craft item",
    "social visit",
    "volunteer",
    "set goal",
    "leisure activity",
    "organize event",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "eat_food",
    "visit_doctor",
    "buy_medicine",
    "take_medicine",
    "social_visit",
    "attend_social_event",
    "pursue_hobby",
    "leisure_activity",
    "work_current_job",
    "change_job",
    "get_educated",
    "trade_goods",
    "buy_goods_from",
    "sell_goods",
    "self_care",
    "attend_therapy",
    "set_goal",
    "clean_up",
    "place_",
    "volunteer_time",
    "vote",
    "make_decision",
    "gather_resource",
    "craft_item",
    "build_structure",
    "repair_item",
    "engage_in_artistic_creation",
    "exercise",
    "meditate",
    "practice_relaxation",
    "invest_wealth",
    "join_community_project",
    "organize_event",
    "cultivate_relationship",
    "eat food",
    "visit doctor",
    "buy medicine",
    "take medicine",
    "social visit",
    "attend social event",
    "pursue hobby",
    "leisure activity",
    "work current job",
    "change job",
    "get educated",
    "trade goods",
    "buy goods from ",
    "sell goods ",
    "self care",
    "attend therapy",
    "set goal",
    "clean up",
    "place ",
    "volunteer time",
    "vote",
    "make decision",
    "gather resource",
    "craft item",
    "build structure",
    "repair item",
    "engage in artistic creation",
    "exercise",
    "meditate",
    "practice relaxation",
    "invest wealth",
    "join community project",
    "organize event",
    "cultivate relationship"
]


empty_results = {}
per_model_result_count = {}
actionable_results = {}
best_time_per_model = {}
stored_results = [{"model_name": "", "responses": [], "best_time_taken": ""}]

for model_name in models_to_test:
    empty_results[model_name] = 0
    per_model_result_count[model_name] = 0
    actionable_results[model_name] = 0
    best_time_per_model[model_name] = 100000

curr_datehour = time.strftime("%Y%m%d-%H%M%S")
filename = f"results{curr_datehour}.txt"

# Test each model and print results
for model_name in models_to_test:
    try:
        results = test_model(model_name, prompts)
        model_responses = []
        print(f"Results: {results} \n")
        for res in results:
            response = res[0]
            time_taken = res[1]
            if response == "" or response == " ":
                response = "No response"
            print(
                f"Model: {model_name}\nResponse: {response}\nProcessing Time: {time_taken} seconds\n\n \n"
            )
            per_model_result_count[model_name] = per_model_result_count[model_name] + 1
            # write to csv
            if response != "No response":
                with open(filename, "a") as f:
                    f.write(f"{model_name},{response},{time_taken}\n\n \n")
                for string in remove_list:
                    response = re.sub(string,'',response)
                if any(keyword.lower() in response.lower() for keyword in actionable_keywords):
                    actionable_results[model_name] += 1
                best_time_per_model[model_name] = min(
                    best_time_per_model[model_name], float(time_taken)
                )
                model_responses.append(response)
            else:
                empty_results[model_name] += 1
            if per_model_result_count[model_name] >= len(prompts):
                
                with open(filename, "a") as f:
                    f.write(
                        f"Model: {model_name}\nEmpty Results: {empty_results[model_name]}\n\n \n"
                    )
                    f.write(
                        f"Model: {model_name}\nActionable Results: {actionable_results[model_name]}\n\n \n"
                    )
                    f.write(
                        f"Model: {model_name}\nBest Time: {best_time_per_model[model_name]}\n\n \n"
                    )
        stored_results.append(
            {
                "model_name": model_name,
                "responses": model_responses,
                "best_time_taken": best_time_per_model[model_name],
            }
        )

    except Exception as e:
        error_line_number = e.__traceback__.tb_lineno
        print(f"Model: {model_name}\nError: {e} Error Line Number: {error_line_number}\n\n \n")
        with open(filename, "a") as f:
            f.write(f"Model: {model_name}\nError: {e}\n\n \n")
        continue
    model_parts = model_name.split("/")
    model_suffix = model_parts[1] if len(model_parts) >= 2 else ""
    with open(filename.split(".txt")[0] + "_" + model_suffix + ".txt", "a") as f:
        f.write(f"Model: {model_name}\n\n \n")
        f.write(f"Best Time: {best_time_per_model[model_name]}\n\n \n")
        f.write(f"Empty Results: {empty_results[model_name]}\n\n \n")
        f.write(f"Actionable Results: {actionable_results[model_name]}\n\n \n")
        f.write(f"Stored Results: {stored_results}\n\n \n")

with open(filename, "a") as f:
    f.write(f"Model with most empty results: {max(empty_results, key=empty_results.get)}\n\n \n")
    f.write(f"Model with most actionable results: {max(actionable_results, key=actionable_results.get)}\n\n \n")
    best_time_values = list(best_time_per_model.values())
    f.write(f"Best Time: {min(best_time_values)} by model {min(best_time_per_model, key=best_time_per_model.get)} for responses such as {stored_results[best_time_values.index(min(best_time_values))]['responses']}\n\n \n")

#Test a. Consistency of results and b. Variability of results
#Test a. Consistency of results
    

def find_keywords(text, keywords):
    found = defaultdict(int)
    for keyword in keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            found[keyword] += 1
    return found

def analyze_responses(keywords, responses):
    analysis_results = []

    keyword_count = defaultdict(int)
    
    for response in responses:
        found_keywords = find_keywords(response, keywords)
        for key, value in found_keywords.items():
            keyword_count[key] += value

    # Calculating variability measures
    counts = keyword_count.values()
    unique_keywords = len(keyword_count)
    most_common_freq = max(counts) if counts else 0
    total_keywords = sum(counts)
    variability = statistics.stdev(counts) if len(counts) > 1 else 0

    analysis_results.append({
        'keyword_frequency': dict(keyword_count),
        'unique_keywords': unique_keywords,
        'most_common_keyword_frequency': most_common_freq,
        'total_keywords': total_keywords,
        'variability': variability,
        'keyword_count': keyword_count
    })

    return analysis_results

# Example usage
class Test:
    def __init__(self, model_name, responses):
        self.model_name = model_name
        self.responses = responses
        self.consistency_scores = self.consistency_of_results()
        self.analysis = self.variability_of_results()
        if self.analysis:
            self.actionability_scores = self.analysis[0]['total_keywords']
            self.unique_keywords = self.analysis[0]['unique_keywords']
            self.most_common_keyword_frequency = self.analysis[0]['most_common_keyword_frequency']
            self.variability_scores = self.analysis[0]['variability']
            self.keyword_frequency = self.analysis[0]['keyword_frequency']
            self.keywords = self.analysis[0]['keyword_count']
        else:
            self.actionability_scores = {}
            self.variability_scores = 0
        self.best_time_taken = best_time_per_model[self.model_name]

    def __repr__(self):
        return f"Test(model_name={self.model_name}, responses={self.responses})"

    def __str__(self):
        return f"Test(model_name={self.model_name}, responses={self.responses})"

    def consistency_of_results(self):
        cosine_scores = []
        sum_similarities = 0
        count = 0
        for i in range(len(self.responses)):
            for j in range(len(self.responses)):
                if i != j:
                    if self.responses[i] == None or self.responses[j] == None or self.responses[i] == "" or self.responses[j] == "":
                        cosine_scores.append(0)
                        print("empty response\n")
                        print(self.responses[i]+ "\n")
                        print(self.responses[j]+ "\n")
                    else:
                        try:
                            cosine_scores.append(cosine(self.responses[i],self.responses[j]))
                        except Exception as e:
                            print(e)
                            print("error\n")
                            print(self.responses[i]+ "\n")
                            print(self.responses[j]+ "\n")
                            cosine_scores.append(0)
        #use cosine_scores to calculate consistency
        n = len(cosine_scores)
        

        for i in range(n):
            for j in range(n):
                if i != j:  # Exclude self-similarities
                    sum_similarities += cosine_scores[i]
                    count += 1
        return sum_similarities / count if count > 0 else 0
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "responses": self.responses,
            "consistency_scores": self.consistency_scores,
            "variability_scores": self.variability_scores,
            "actionability_scores": self.actionability_scores,
            "best_time_taken": self.best_time_taken,
            "unique_keywords": getattr(self, 'unique_keywords', None),
            "most_common_keyword_frequency": getattr(self, 'most_common_keyword_frequency', None),
            "keyword_frequency": getattr(self, 'keyword_frequency', None),
            "keywords": getattr(self, 'keywords', None)
        }
    #Test b. Variability of results

    def variability_of_results(self):
        keywords = actionable_keywords
        analysis_results = analyze_responses(keywords,self.responses)
        return analysis_results

    #Test c. Actionability of results
tests = []
for model in models_to_test:
    dict_with_value = next((item for item in stored_results if item.get("model_name") == model), None)
    test = Test(model, dict_with_value['responses'])
    print(test)
    print(f"Consistency of results: {test.consistency_scores}")
    print(f"Variability of results: {test.variability_scores}")
    print(f"Actionability of results: {test.actionability_scores}")
    print(f"Unique keywords: {test.unique_keywords}")
    print(f"Most common keyword frequency: {test.most_common_keyword_frequency}")
    print(f"Keyword frequency: {test.keyword_frequency}")
    print(f"Keywords: {test.keywords}")
    print(f"Best time taken: {test.best_time_taken}")
    print("\n")
    test_data = test.to_dict()

    model_parts = model.split("/")
    model_suffix = model_parts[1] if len(model_parts) >= 2 else model.replace("/", "")
    filename_suffix = filename.split(".txt")[0] + "_" + model_suffix + ".json"
    tests.append(test)
    with open(filename_suffix, "a") as f:
        f.write(f"Consistency of results: {test.consistency_scores}\n\n \n")
        f.write(f"Variability of results: {test.variability_scores}\n\n \n")
        f.write(f"Actionability of results: {test.actionability_scores}\n\n \n")
        f.write(f"Unique keywords: {test.unique_keywords}\n\n \n")
        f.write(f"Most common keyword frequency: {test.most_common_keyword_frequency}\n\n \n")
        f.write(f"Keyword frequency: {test.keyword_frequency}\n\n \n")
        f.write(f"Keywords: {test.keywords}\n\n \n")
        f.write(f"Best time taken: {test.best_time_taken}\n\n \n")
        f.write("\n\n\n")
        f.write("Full Details:\n\n \n")
        f.write(f"{test}\n\n \n")
    with open(filename_suffix, "w") as f:
        json.dump(test_data, f, indent=4)

def get_highest_attribute(tests, attribute):
    max_test = None
    for test in tests:
        test_attribute = getattr(test, attribute)
        if max_test is None or test_attribute > getattr(max_test, attribute):
            max_test = test
    return {max_test.model_name: getattr(max_test, attribute)} if max_test else {}

def get_lowest_attribute(tests, attribute):
    min_test = None
    for test in tests:
        test_attribute = getattr(test, attribute)
        if min_test is None or test_attribute < getattr(min_test, attribute):
            min_test = test
    return {min_test.model_name: getattr(min_test, attribute)} if min_test else {}

# Usage:
# highest_consistency = get_highest_attribute(tests, 'consistency_scores')
highest_consistency = get_highest_attribute(tests, 'consistency_scores')
highest_variability = get_highest_attribute(tests, 'variability_scores')
highest_actionability = get_highest_attribute(tests, 'actionability_scores')
lowest_consistency = get_lowest_attribute(tests, 'consistency_scores')
lowest_variability = get_lowest_attribute(tests, 'variability_scores')
lowest_actionability = get_lowest_attribute(tests, 'actionability_scores')
best_time_taken = get_lowest_attribute(tests, 'best_time_taken')
slowest_time_taken = get_highest_attribute(tests, 'best_time_taken')
print(f"Highest consistency: {highest_consistency}")
print(f"Highest variability: {highest_variability}")
print(f"Highest actionability: {highest_actionability}")
print(f"Lowest consistency: {lowest_consistency}")
print(f"Lowest variability: {lowest_variability}")
print(f"Lowest actionability: {lowest_actionability}")
print(f"Best time taken: {best_time_taken}")
print(f"Slowest time taken: {slowest_time_taken}")

summary_dict = {
    "highest_consistency": highest_consistency,
    "highest_variability": highest_variability,
    "highest_actionability": highest_actionability,
    "lowest_consistency": lowest_consistency,
    "lowest_variability": lowest_variability,
    "lowest_actionability": lowest_actionability,
    "best_time_taken": best_time_taken,
    "slowest_time_taken": slowest_time_taken
}
summary_filename = filename.split(".txt")[0] + "_" + "final_summary" + ".json"

with open(summary_filename, "w") as f:
    json.dump(summary_dict, f, indent=4)
    
