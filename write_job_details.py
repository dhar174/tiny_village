from gettext import find
import logging
from math import log
import shutil
import sys
from llama_cpp import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
    Llama,
    StoppingCriteriaList,
    StoppingCriteria,
)
import json
from fuzzywuzzy import fuzz, process
import regex
from sympy import N, det
from torch import ge

# Define possible motives and salary range constraints
possible_motives = [
    "hunger motive",
    "wealth motive",
    "mental health motive",
    "social wellbeing motive",
    "happiness motive",
    "health motive",
    "shelter motive",
    "stability motive",
    "luxury motive",
    "hope motive",
    "success motive",
    "control motive",
    "job performance motive",
    "beauty motive",
    "community motive",
    "material goods motive",
    "family motive",
]
salary_range = {"min": 10000, "max": 200000}


# Function to get the width of the terminal
def get_terminal_width():
    return shutil.get_terminal_size((80, 20)).columns


# Function to print progress with line wrapping, avoiding mid-word breaks
def print_progress(text):
    terminal_width = get_terminal_width()
    # if "\n" in text:

    # Break the text into lines that fit within the terminal

    sys.stdout.writelines(text + "|" + "\r")
    sys.stdout.flush()


def clear_line():
    # Clear the current line in the terminal
    sys.stdout.write("\r" + " " * get_terminal_width() + "\r")


def sanitize_input(input_str, also_remove=[]):
    # Replace new lines with spaces or any desired character
    input_str = str(input_str.strip())
    for item in also_remove:
        input_str = input_str.replace(item, "")
    return input_str.replace("\n", " ")


logging.basicConfig(level=logging.DEBUG)
# Load the quantized Llama 3 model
model_path = "/mnt/d/Cache_for_AI/meta-llama-3.1-8b-instruct.Q4_K_M.gguf"  # Update this with the actual path to your GGUF model file
llama = Llama(model_path, use_mlock=True, n_ctx=1024)


# Test the model with a sample prompt in utf-8 bytes
# test_prompt = llama.tokenize("What is the role of an Account Manager?".encode("utf-8"))
# response = llama(test_prompt)
# logging.info(f"Test prompt: {llama.detokenize(test_prompt)}\nResponse: {response}")

# # Test with generation function instead of direct response
# logging.info("Testing with generation function...")
# response_text = ""
# token_count = 0

# end_token_id = llama.tokenize(b"</s>")[0]
# question_token_id = llama.tokenize(b"?")[0]
# newline_token_id = llama.tokenize(b"\n")[0]

# detokens = []

# try:
#     for token in llama.generate(
#         test_prompt,
#         temp=0.8,
#         stopping_criteria=StoppingCriteriaList(
#             [
#                 lambda input_ids, logits: len(input_ids) >= 100,
#                 lambda input_ids, logits: token_count >= 100,
#                 # lambda input_ids, logits: end_token_id in input_ids,
#                 # lambda input_ids, logits: question_token_id in input_ids,
#                 # lambda input_ids, logits: newline_token_id in input_ids,
#             ]
#         ),
#     ):
#         # Print the status, followed by a carriage return
#         token_count += 1
#         detoken = llama.detokenize([token])
#         detoken = detoken.decode("utf-8")
#         detokens.append(detoken)

#         response_text += detoken
#         terminal_width = get_terminal_width()
#         if "\n" in detoken:
#             # Break the text into lines that fit within the terminal
#             lines = response_text.split("\n")
#             for line in lines[:-1]:
#                 print_progress("\n" + line)
#                 clear_line()

#                 response_text = lines[-1]
#         # clear_line()
#         if len(response_text) > terminal_width:
#             # Find the last space within the terminal width
#             break_point = response_text.rfind(" ", 0, terminal_width)
#             if break_point == -1:  # No space found, force break at previous break point
#                 break_point = terminal_width
#             # Print up to the break point
#             print_progress(response_text[:break_point] + "..." + "\n")
#             response_text = response_text[break_point:].lstrip()
#         # # Print up to the break point
#         # # sys.stdout.writelines(response_text[:break_point] + "..." + "\n")
#         # response_text = response_text[
#         #     break_point:
#         # ].lstrip()  # Remove leading spaces
#         # sys.stdout.write("\r" + " " * break_point)  # Clear the line
#         # sys.stdout.flush()
#         # Print the remaining text
#         else:
#             print_progress(response_text)


# except Exception as e:
#     logging.error(
#         f"Error in generation: {e} token count: {token_count} detokens: {detokens}"
#     )

# logging.info(f"\nToken count: {token_count}")


# # Print the final response
# logging.info(
#     f"\nTest prompt: {llama.detokenize(test_prompt)}\nResponse: {response_text}"
# )
# # pause a few seconds
# import time

# time.sleep(5)

# # Test as a chat completion
# logging.info("Testing chat completion...")
chat_message = [
    {
        "role": "system",
        "content": "You are an AI assistant that writes job descriptions and other job-related content in JSON format.",
    },
    {
        "role": "user",
        "content": f"Given a job title, generate a job description, \n"
        f"salary, required skills, education, experience, and character motives. \n"
        f"Ensure the salary is within the range $${salary_range['min']} to $${salary_range['max']} \n"
        f"and the motives are chosen from the following list: {possible_motives}. \n"
        f"Please respond in JSON format, like the following example: \n"
        f"{{'Clown':{{'title':'Clown','description': 'A clown is a performer who uses comedy and physicality to entertain audiences.', 'salary':'$30,000','skills':['Comedy','Physicality'],'education':'High school diploma','experience':'1-2 years','associated character motives':['happiness motive','social wellbeing motive']}}}}",
    },
    {
        "role": "assistant",
        "content": f"Sure, I can help with that. I will start by generating a job description for an Account Manager, including the salary, required skills, education, experience, and character motives. I will respond in JSON format. Let's begin. \n"
        f"{{'Account Manager': {{'title':'Account Manager','description':'An account manager is responsible for managing relationships with clients, ensuring their needs are met, and identifying new business opportunities.','salary':'$60,000','skills':['Communication','Negotiation','Problem-solving'],'education':'Bachelors degree','experience':'2-3 years','associated character motives':['wealth motive','success motive','job performance motive']}}}} \n"
        f"How was that? If you would like me to adjust any details, please let me know.",
    },
]
# llama.reset()
# response = llama.create_chat_completion(chat_message)
# logging.info(f"Test prompt: {test_prompt}\nResponse: {response}")
# time.sleep(5)

# # Test as a regular completion
# logging.info("Testing regular completion...")
# response = llama.create_completion(chat_message[1]["content"])
# logging.info(f"Test prompt: {test_prompt}\nResponse: {response}")
# time.sleep(5)
# # Test with stream mode one
# try:
#     for response in llama.create_chat_completion(
#         chat_message, temperature=0.8, stream=True
#     ):
#         logging.info(
#             f"Test prompt: {test_prompt}\nResponse: {llama.detokenize(response)}"
#         )
#     logging.info(f"Test prompt: {test_prompt}\nResponse: {llama.detokenize(response)}")
# except Exception as e:
#     logging.error(f"Error in stream mode 1: {e}")

# # print the context of the model
# logging.info(f"Model context: {llama.ctx}\n")
# # logging.info(f"Model context tokens: {llama.detokenize(llama.ctx)}\n")
# logging.info(f"Model cache: {llama.cache}\n")
# logging.info("Test complete.")

# Load job data from JSON file
try:
    with open("sorted_jobs.json", "r") as file:
        jobs_data = json.load(file)
except FileNotFoundError:
    logging.error(
        "Please ensure the jobs.json file is present in the current directory."
    )
    jobs_data = {"jobs": {}}


# Function to find the closest valid motive
def map_to_valid_motive(generated_motive, valid_motives, threshold=80):
    closest_match, score = process.extractOne(
        generated_motive, valid_motives, scorer=fuzz.token_set_ratio
    )
    if score >= threshold:
        return closest_match
    return None


def compare_words(word1, word2, threshold=80):
    return fuzz.token_set_ratio(word1, word2) >= threshold


def find_similiar_word_in_text(word, text, threshold=80):
    # Returns a boolean value indicating if the word is in the text
    return (
        process.extractOne(word, text.split(), scorer=fuzz.token_set_ratio)[1]
        >= threshold
    )


def fallback_extraction(response_text, job_title):
    try:
        logging.info(f"Trying to decode response without JSON decoding using regex")
        generated_data = {}
        words = regex.findall(r"\b\w+\b", response_text)

        try:

            similiar_word = process.extractOne(
                "title", words, scorer=fuzz.token_set_ratio
            )[0]
            logging.info(f"Similiar word for title: {similiar_word}")
            reg = regex.findall(
                r"(?:title|job title|\b"
                + similiar_word
                + r"\b)(.*?)(?:\n|description)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("title", response_text) and reg not in [
                None,
                [],
            ]:
                # find any text (by the sentence) after the word title or job title, up to the next newline or the word description

                generated_data.update({"title": reg[0].strip()})
        except Exception as e:
            logging.error(
                f"Error processing job title for job {job_title} in fallback extraction: {e}, {response_text}, regex: {reg} and words: {words}"
            )

        try:
            similiar_word = process.extractOne(
                "description",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            sim_end_word = process.extractOne(
                "salary",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for description: {similiar_word}")
            logging.info(f"Similiar word for salary: {sim_end_word}")
            reg = regex.findall(
                r"(?:description|\b"
                + similiar_word
                + r"\b)(.*?)(?:\n|salary|\b"
                + sim_end_word
                + r"\b)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("description", response_text) and reg not in [
                None,
                [],
            ]:
                # find any text (by the sentence) after the word description, up to the next newline or the word salary

                generated_data.update({"description": reg[0].strip()})
        except Exception as e:
            logging.error(
                f"Error processing job description for job {job_title} in fallback extraction: {e}, {response_text}, regex: {reg} and words: {words}"
            )
        try:
            similiar_word = process.extractOne(
                "salary",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for salary: {similiar_word}")
            reg = regex.search(
                r"(?:^|\.)\s*(?:[^.]*?\b(?:salary|pay|\b"
                + similiar_word
                + r"\b)\b[^.]*?(\d[\d,]*)[^.]*?\.)",
                response_text,
                regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("salary", response_text) and reg not in [
                None,
                [],
            ]:

                # find any numbers after the word salary
                generated_data.update({"salary": int(reg.group(1).replace(",", ""))})
        except Exception as e:
            logging.error(
                f"Error processing job salary for job {job_title} in fallback extraction: {e}, {response_text}, regex: {reg} and words: {words}"
            )
        try:
            similiar_word = process.extractOne(
                "skills",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for skills: {similiar_word}")
            reg = regex.findall(
                r"(?:skills|\b" + similiar_word + r"\b)(.*?)(?:\n)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
                overlapped=False,
            )
            if find_similiar_word_in_text("skills", response_text) and reg not in [
                None,
                [],
            ]:

                # find any text (by the sentence) after the word skills. Could be a list [] of skills separated by commas
                generated_data.update(
                    {"skills": [skill.strip() for skill in reg[0].strip().split(",")]}
                )
        except Exception as e:
            logging.error(
                f"Error processing job skills for job {job_title} in fallback extraction: {e}, {response_text}, with regex: {reg} and words: {words}"
            )
        try:
            similiar_word = process.extractOne(
                "education",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for education: {similiar_word}")
            reg = regex.findall(
                r"(?:education|\b" + similiar_word + r"\b)(.*?)(?:\n)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("education", response_text) and reg not in [
                None,
                [],
            ]:

                # find any text (by the sentence) after the word education, up to the next newline
                generated_data.update({"education": reg[0].strip()})
        except Exception as e:
            logging.error(
                f"Error processing job education for job {job_title} in fallback extraction: {e}, {response_text} with regex: {reg} and words: {words}"
            )

        try:
            similiar_word = process.extractOne(
                "experience",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for experience: {similiar_word}")
            reg = regex.search(
                r"(?:experience|\b" + similiar_word + r"\b)(.*?)(\d+)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("experience", response_text) and reg not in [
                None,
                [],
            ]:

                # find any numbers after the word experience
                generated_data.update({"experience": int(reg.group(2))})
        except Exception as e:
            logging.error(
                f"Error processing job experience for job {job_title} in fallback extraction: {e}, {response_text} with regex: {reg} and words: {words}"
            )
        try:
            similiar_word = process.extractOne(
                "motives",
                words,
                scorer=fuzz.token_set_ratio,
            )[0]
            logging.info(f"Similiar word for motives: {similiar_word}")
            reg = regex.findall(
                r"(?:motives|\b" + similiar_word + r"\b)(.*?)(?:\n)",
                response_text,
                flags=regex.IGNORECASE | regex.DOTALL,
            )
            if find_similiar_word_in_text("motives", response_text) and reg not in [
                None,
                [],
            ]:
                # find any text (by the sentence) after the word motives, up to the next newline

                generated_data.update(
                    {
                        "associated character motives": [
                            motive.strip() for motive in reg[0].strip().split(",")
                        ]
                    }
                )
        except Exception as e:
            logging.error(
                f"Error processing job motives for job {job_title} in fallback extraction: {e}, {response_text} with regex: {reg} and words: {words}"
            )

    except Exception as e:
        logging.error(
            f"Error processing job {job_title} in fallback extraction: {e}, {response_text}. Data generated so far: {generated_data}"
        )
    return generated_data


# Function to ensure constraints are met
def enforce_constraints(data):
    # Validate and adjust salary
    salary = int(data.get("salary", salary_range["min"]))
    salary = max(min(salary, salary_range["max"]), salary_range["min"])
    data["salary"] = f"${salary:,}"

    # Map and filter valid motives
    filtered_motives = []
    for motive in data.get("associated character motives", []):
        mapped_motive = map_to_valid_motive(motive, possible_motives)
        if mapped_motive:
            filtered_motives.append(mapped_motive)
    data["associated character motives"] = filtered_motives

    return data


def generate_job_details():
    # Process each job
    new_chat = []
    output_data = {"jobs": {}}
    for job_title, job_details in jobs_data["jobs"].items():
        new_chat = chat_message.copy()
        new_chat.append(
            {
                "role": "user",
                "content": f"Great job! Here's the next one: {job_title}\n"
                f"Please respond in JSON format with the job description, salary, required skills, education, experience, and character motives.",
            }
        )
        logging.info(f"Processing job: {job_title}")
        try:
            llama.reset()
            generated_response = llama.create_chat_completion(
                new_chat, max_tokens=256, temperature=0.8
            )
            try:
                logging.info(
                    f"Generated response: {generated_response['choices'][0]['message']['content']}"
                )
                # split the response to get the json part. Use regex to get the json part. Count the number of curly braces to know if the json is complete
                curly_braces = 0
                last_brace = None
                all_curly_braces = None
                for char, i in enumerate(
                    generated_response["choices"][0]["message"]["content"]
                ):
                    if char == "{":
                        curly_braces += 1
                        last_brace = i
                    elif char == "}":
                        curly_braces -= 1
                        last_brace = i
                    if curly_braces == 0:
                        break
                if last_brace is None:
                    last_brace = len(
                        generated_response["choices"][0]["message"]["content"]
                    )
                if curly_braces == 0:
                    all_curly_braces = regex.search(
                        r"\{(?:[^{}]+|(?R))*\}",
                        generated_response["choices"][0]["message"]["content"],
                    )
                    if all_curly_braces is not None:
                        try:
                            if isinstance(all_curly_braces, str):
                                all_curly_braces = all_curly_braces.replace("'", '"')
                                generated_data = json.loads(all_curly_braces)
                            elif isinstance(all_curly_braces, regex.Match):
                                all_curly_braces = all_curly_braces.group().replace(
                                    "'", '"'
                                )
                                generated_data = json.loads(all_curly_braces)

                        except json.JSONDecodeError as jsnerr:
                            logging.error(f"JSON decoding error: {jsnerr}")
                            logging.error(
                                f"Problematic JSON string: {all_curly_braces}"
                            )
                elif last_brace:
                    # Not all curly braces are closed. Find the last occurence of a curly brace and get the json part from the beginning to that point
                    # We can then send the json part to the json decoder and then send the rest of the text to the fallback extraction function
                    all_curly_braces = regex.search(
                        r"\{(?:[^{}]+|(?R))*\}",
                        generated_response["choices"][0]["message"]["content"][
                            :last_brace
                        ],
                    )
                    after_curly_braces = generated_response["choices"][0]["message"][
                        "content"
                    ][last_brace:]

                    fallback_data = fallback_extraction(after_curly_braces, job_title)
                    if all_curly_braces is not None:
                        generated_data = json.loads(all_curly_braces)
                        logging.info(
                            f"Data generated from all_curly_braces: {generated_data}"
                        )
                    logging.info(f"Fallback data: {fallback_data}")

                else:
                    generated_data = fallback_extraction(
                        generated_response["choices"][0]["message"]["content"],
                        job_title,
                    )
                if generated_data is not None or generated_data != {}:
                    logging.info(f"First-pass generated data: {generated_data} \n")
                else:
                    data = regex.search(
                        r"\{(?:[^{}]+|(?R))*\}",
                        generated_response["choices"][0]["message"]["content"],
                    )
                    if data is not None:
                        try:
                            if isinstance(data, str):
                                data = data.replace("'", '"')
                                generated_data = json.loads(data)
                            elif isinstance(data, regex.Match):
                                data = data.group().replace("'", '"')
                                generated_data = json.loads(data)

                        except json.JSONDecodeError as jsnerr:
                            logging.error(f"JSON decoding error: {jsnerr}")
                            logging.error(f"Problematic JSON string: {data}")

            except Exception as e:
                logging.error(
                    f"Error decoding generated response for job {job_title}: {generated_response}, {e}"
                )
                generated_data = fallback_extraction(
                    generated_response["choices"][0]["message"]["content"], job_title
                )

            if generated_data is not None or generated_data != {}:
                logging.info(f"Final generated data: {generated_data} \n")

            # Ensure constraints are met
            constrained_data = enforce_constraints(generated_data)

            # Append to output data
            output_data["jobs"][job_title] = constrained_data

            # Save the result incrementally
            with open("output_jobs.json", "w") as outfile:
                json.dump(output_data, outfile, indent=2)
            logging.info(f"Saved output data for job {job_title} to output_jobs.json")
        except Exception as e:
            logging.error(
                f"Error processing job {job_title}: {e}, {generated_response}"
            )

    return True if output_data else False


if __name__ == "__main__":
    logging.info("Beginning job details generation...")
    done = generate_job_details()
    if done:
        logging.info("Job details generation complete.")
    else:
        logging.error("Job details generation failed.")
