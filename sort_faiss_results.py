from itertools import combinations

import pandas as pd


# Load the latest data
latest_file_path = "results.txt"
with open(latest_file_path, "r") as file:
    latest_data = file.read()


# Function to parse the updated data
def parse_updated_results(data):
    # Initialize parsing structures
    categories = {}
    lines = data.split("\n")
    current_category = None
    current_query = None
    capturing = False
    query_results = []

    # Process each line
    for line in lines:
        line = line.strip()
        if line.startswith("Results for queries with"):
            if current_category and query_results:
                # Save the last set of results before switching categories
                categories[current_category]["queries"][current_query] = query_results
            # Determine category
            current_category = line
            categories[current_category] = {"type": None, "queries": {}}
            query_results = []
        elif "Index type:" in line and "build count" not in line:
            # Capture index type
            categories[current_category]["type"] = line.split("Index type: ")[
                -1
            ].strip()
            print(f"Current index type: {categories[current_category]['type']}")
            assert categories[current_category]["type"] in [
                "IndexFlatL2",
                "IndexFlatIP",
            ]
        elif line.startswith("Query:"):
            if current_query and query_results:
                # Save previous query results
                categories[current_category]["queries"][current_query] = query_results
            # New query
            current_query = line.split("Query:")[-1].strip()
            query_results = []
            capturing = (
                False  # Reset capturing until we find 'results from search memories:'
            )
        elif "results from search memories:" in line:
            capturing = True
        elif capturing and "," in line and "weight:" in line:
            # Capture results data
            result, weight = line.rsplit(",", 1)
            weight = float(weight.split("weight:")[-1].strip())
            query_results.append((result.strip(), weight))

    # Store the last query's results
    if current_category and current_query and query_results:
        categories[current_category]["queries"][current_query] = query_results

    return categories


# Parse the updated data
parsed_updated_results = parse_updated_results(latest_data)
# print(f"Parsed updated results: {parsed_updated_results}")


# Define a function to compare results across categories
def compare_results_across_categories(results):
    comparison_data = []
    queries = set()
    for category in results:
        queries.update(results[category]["queries"].keys())

    for query in queries:
        all_results = []
        for category, data in results.items():
            # print(f"Category: {category} \n \n")
            query_data = data["queries"].get(query, [])
            if query_data:  # Check if query_data is not empty
                sorted_query_data = sorted(
                    query_data,
                    key=lambda x: x[1],
                    reverse=(
                        data["type"].lower() != "indexflatl2" if data["type"] else False
                    ),
                )
                if sorted_query_data:
                    all_results.append((category, sorted_query_data[0]))
                else:
                    all_results.append((category, None))

        if len(all_results) > 1:
            for (cat1, res1), (cat2, res2) in combinations(all_results, 2):
                if (
                    res1 and res2 and res1 != res2
                ):  # Check if res1 and res2 are not empty
                    comparison_data.append([query, cat1, res1, cat2, res2])

    return comparison_data


# Execute comparison
final_comparison_data = compare_results_across_categories(parsed_updated_results)

# Convert to DataFrame
final_comparison_df = pd.DataFrame(
    final_comparison_data,
    columns=["Query", "Category 1", "Top Result 1", "Category 2", "Top Result 2"],
)

latest_output_path = "latest_comparison_results.csv"
final_comparison_df.to_csv(latest_output_path, index=False)

print(f"Latest comparison results saved to: {latest_output_path}")
