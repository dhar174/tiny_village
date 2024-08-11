import json
import logging


class JobRoles:
    def __init__(
        self,
        job_name,
        job_title,
        job_description,
        job_salary,
        job_skills,
        job_education,
        req_job_experience,
        job_motives,
        location=None,
    ):
        self.job_name = self.set_job_name(job_name)
        self.job_title = job_title
        self.job_description = self.set_job_description(job_description)
        self.job_salary = self.set_job_salary(job_salary)
        self.job_skills = self.set_job_skills(job_skills)
        self.job_education = self.set_job_education(job_education)
        self.req_job_experience = self.set_job_experience(req_job_experience)
        self.job_motives = self.set_job_motives(job_motives)
        self.location = location

    def __repr__(self):
        return f"JobRoles({self.job_name}, {self.job_description}, {self.job_salary}, {self.job_skills}, {self.job_education}, {self.req_job_experience}, {self.job_motives})"

    def __str__(self):
        return f"JobRoles with name {self.job_name}, description {self.job_description}, salary {self.job_salary}, skills {self.job_skills}, education {self.job_education}, experience {self.req_job_experience}, motives {self.job_motives}."

    def __eq__(self, other):
        if not isinstance(other, JobRoles):
            return False
        return (
            self.job_name == other.job_name
            and self.job_description == other.job_description
            and self.job_salary == other.job_salary
            and self.job_skills == other.job_skills
            and self.job_education == other.job_education
            and self.req_job_experience == other.req_job_experience
            and self.job_motives == other.job_motives
        )

    def hash_nested_list(self, obj):
        try:
            if isinstance(obj, list):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(
                    (key, self.hash_nested_list(value)) for key, value in obj.items()
                )
            elif isinstance(obj, set):
                return frozenset(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, tuple):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif hasattr(obj, "__hash__") and callable(getattr(obj, "__hash__")):
                # Test if the object can be hashed without raising an error
                try:
                    hash(obj)
                    return obj
                except TypeError:
                    if hasattr(obj, "__dict__"):
                        return tuple(
                            (key, self.hash_nested_list(value))
                            for key, value in obj.__dict__.items()
                        )
                    else:
                        # If the object is not hashable and has no __dict__, return its id or a string representation
                        return id(obj)
            elif hasattr(obj, "__dict__"):  # For custom objects without __hash__ method
                return tuple(
                    (key, self.hash_nested_list(value))
                    for key, value in obj.__dict__.items()
                )
            else:
                return obj
        except Exception as e:
            logging.error(f"Error hashing object: {e}")
            return None

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.job_name,
                    self.job_description,
                    self.job_salary,
                    tuple(self.job_skills),
                    self.job_education,
                    self.req_job_experience,
                    tuple(self.job_motives),
                ]
            )
        )

    def get_job_name(self):
        return self.job_name

    def set_job_name(self, job_name):
        # Warning: Name MUST be unique! Check for duplicates before setting.

        self.job_name = job_name
        return self.job_name

    def get_job_description(self):
        return self.job_description

    def set_job_description(self, job_description):
        self.job_description = job_description
        return self.job_description

    def get_job_salary(self):
        return self.job_salary

    def set_job_salary(self, job_salary):
        self.job_salary = job_salary
        return self.job_salary

    def get_job_skills(self):
        return self.job_skills

    def set_job_skills(self, job_skills):
        self.job_skills = job_skills
        return self.job_skills

    def get_job_education(self):
        return self.job_education

    def set_job_education(self, job_education):
        self.job_education = job_education
        return self.job_education

    def get_job_experience(self):
        return self.req_job_experience

    def set_job_experience(self, req_job_experience):
        self.req_job_experience = req_job_experience
        return self.req_job_experience

    def get_job_motives(self):
        return self.job_motives

    def set_job_motives(self, job_motives):
        self.job_motives = job_motives
        return self.job_motives

    def to_dict(self):
        return {
            "name": self.job_name,
            "description": self.job_description,
            "salary": self.job_salary,
            "skills": self.job_skills,
            "education": self.job_education,
            "experience": self.req_job_experience,
            "motives": self.job_motives,
        }


class JobRules:
    def __init__(self):
        job_roles = json.load(open("job_roles.json"))
        job_count = len([job_roles["jobs"][job_role] for job_role in job_roles["jobs"]])
        logging.debug(f"Job count: {job_count}")
        if job_count != len(
            [
                job_roles["jobs"][job_role][jobname]
                for job_role in job_roles["jobs"]
                for jobname in job_roles["jobs"][job_role]
                if job_role.lower() == jobname.lower()
            ]
        ):
            logging.error(
                f"Not all job roles parsed correctly, counted {job_count} but parsed {len([job_roles['jobs'][job_role][jobname] for job_role in job_roles['jobs'] for jobname in job_roles['jobs'][job_role] if job_role.lower() == jobname.lower()])}"
            )
            if job_count > len(
                [
                    job_roles["jobs"][job_role][jobname]
                    for job_role in job_roles["jobs"]
                    for jobname in job_roles["jobs"][job_role]
                    if job_role.lower() == jobname.lower()
                ]
            ):
                logging.error(
                    f"Missing job roles: {[job_role for job_role in job_roles['jobs'] if job_role not in [jobname for job_role in job_roles['jobs'] for jobname in job_roles['jobs'][job_role] if job_role.lower() == jobname.lower()]]}"
                )
            elif job_count < len(
                [
                    job_roles["jobs"][job_role][jobname]
                    for job_role in job_roles["jobs"]
                    for jobname in job_roles["jobs"][job_role]
                    if job_role.lower() == jobname.lower()
                ]
            ):
                logging.error(
                    f"Extra job roles: {[jobname for job_role in job_roles['jobs'] for jobname in job_roles['jobs'][job_role] if job_role.lower() == jobname.lower() and jobname not in [job_role for job_role in job_roles['jobs']]]}"
                )
        logging.debug(
            f"Job roles: {[job_roles['jobs'][job_role][jobname] for job_role in job_roles['jobs'] for jobname in job_roles['jobs'][job_role] if ((job_role.lower() == jobname.lower() or job_role.lower() in jobname.lower())or (jobname.lower() == job_role.lower() or jobname.lower() in job_role.lower()))]}"
        )
        self.ValidJobRoles = [
            JobRoles(
                key, title, description, salary, skills, education, experience, motives
            )
            for job_role in job_roles["jobs"]
            for key, job_data in job_roles["jobs"][job_role].items()
            if isinstance(job_data, dict)
            and (
                (job_role.lower() == key.lower() or job_role.lower() in key.lower())
                or (key.lower() == job_role.lower() or key.lower() in job_role.lower())
            )
            for title, description, salary, skills, education, experience, motives in [
                job_data.values()
            ]
        ]
        logging.debug(f"Valid job roles: {self.ValidJobRoles}")
        logging.debug(
            f"Found: {[job_role.get_job_name() for job_role in self.ValidJobRoles]}"
        )

    def __repr__(self):
        return f"JobRules({self.ValidJobRoles})"

    def __str__(self):
        return f"JobRules with valid job roles {self.ValidJobRoles}."

    def __eq__(self, other):
        if not isinstance(other, JobRules):
            if isinstance(other, list):
                return self.ValidJobRoles == other
            else:
                return False
        return self.ValidJobRoles == other.ValidJobRoles

    def check_job_role_validity(self, job_role: JobRoles):
        if job_role in self.ValidJobRoles:
            return True
        else:
            return False

    def check_job_name_validity(self, job_name: str):
        for job_role in self.ValidJobRoles:
            if (
                job_role.get_job_name() == job_name
                or job_name in job_role.get_job_name()
                or job_role.get_job_name() in job_name
                or job_name.lower() == job_role.job_title.lower()
                or job_role.job_title.lower() in job_name.lower()
                or job_name.lower() in job_role.job_title.lower()
            ):

                return True
        return False


# Job class is a subclass of JobRoles and inherits from it
class Job(JobRoles):
    def __init__(
        self,
        job_name,
        job_description,
        job_salary,
        job_skills,
        job_education,
        req_job_experience,
        job_motives,
        job_title="",
        location=None,
    ):
        super().__init__(
            job_name,
            job_title,
            job_description,
            job_salary,
            job_skills,
            job_education,
            req_job_experience,
            job_motives,
            location,
        )
        # Warning: Name MUST be unique! Check for duplicates before setting.

        self.job_name = job_name
        self.job_description = job_description
        self.job_salary = job_salary
        self.job_skills = job_skills
        self.job_education = job_education
        self.req_job_experience = req_job_experience
        self.job_motives = job_motives
        self.available = True
        self.job_title = job_title
        self.location = location

    def __repr__(self):
        return f"Job({self.job_name}, {self.job_description}, {self.job_salary}, {self.job_skills}, {self.job_education}, {self.req_job_experience}, {self.job_motives})"

    def __str__(self):
        return f"Job with name {self.job_name}, description {self.job_description}, salary {self.job_salary}, skills {self.job_skills}, education {self.job_education}, experience {self.req_job_experience}, motives {self.job_motives}."

    def __eq__(self, other):
        if not isinstance(other, Job) or not isinstance(other, JobRoles):
            return False
        return (
            self.job_name == other.job_name
            and self.job_description == other.job_description
            and self.job_salary == other.job_salary
            and self.job_skills == other.job_skills
            and self.job_education == other.job_education
            and self.req_job_experience == other.req_job_experience
            and self.job_motives == other.job_motives
            and self.available == other.available
            and self.job_title == other.job_title
            and self.location == other.location
        )

    def hash_nested_list(self, obj):
        try:
            if isinstance(obj, list):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(
                    (key, self.hash_nested_list(value)) for key, value in obj.items()
                )
            elif isinstance(obj, set):
                return frozenset(self.hash_nested_list(item) for item in obj)
            elif isinstance(obj, tuple):
                return tuple(self.hash_nested_list(item) for item in obj)
            elif hasattr(obj, "__hash__") and callable(getattr(obj, "__hash__")):
                # Test if the object can be hashed without raising an error
                try:
                    hash(obj)
                    return obj
                except TypeError:
                    if hasattr(obj, "__dict__"):
                        return tuple(
                            (key, self.hash_nested_list(value))
                            for key, value in obj.__dict__.items()
                        )
                    else:
                        # If the object is not hashable and has no __dict__, return its id or a string representation
                        return id(obj)
            elif hasattr(obj, "__dict__"):  # For custom objects without __hash__ method
                return tuple(
                    (key, self.hash_nested_list(value))
                    for key, value in obj.__dict__.items()
                )
            else:
                return obj
        except Exception as e:
            logging.error(f"Error hashing object: {e}")
            return None

    def __hash__(self):
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(e) for e in obj)
            elif isinstance(obj, set):
                return frozenset(make_hashable(e) for e in obj)
            return obj

        return hash(
            tuple(
                [
                    self.job_name,
                    self.job_description,
                    self.job_salary,
                    make_hashable(self.job_skills),
                    self.job_education,
                    self.req_job_experience,
                    make_hashable(self.job_motives),
                    self.available,
                    self.job_title,
                    make_hashable(self.location),
                ]
            )
        )

    def get_job_name(self):
        return self.job_name

    def set_job_name(self, job_name):
        # Warning: Name MUST be unique! Check for duplicates before setting.

        self.job_name = job_name
        return self.job_name

    def get_job_description(self):
        return self.job_description

    def set_job_description(self, job_description):
        self.job_description = job_description
        return self.job_description

    def get_job_salary(self):
        return self.job_salary

    def set_job_salary(self, job_salary):
        self.job_salary = job_salary
        return self.job_salary

    def get_job_skills(self):
        return self.job_skills

    def set_job_skills(self, job_skills):
        self.job_skills = job_skills
        return self.job_skills

    def get_job_education(self):
        return self.job_education

    def set_job_education(self, job_education):
        self.job_education = job_education
        return self.job_education

    def to_dict(self):
        return {
            "name": self.job_name,
            "description": self.job_description,
            "salary": self.job_salary,
            "skills": self.job_skills,
            "education": self.job_education,
            "experience": self.req_job_experience,
            "motives": self.job_motives,
        }


class JobManager:
    def __init__(self):
        self.job_rules = JobRules()

    def __repr__(self):
        return f"JobManager({self.job_rules})"

    def __str__(self):
        return f"JobManager with job rules {self.job_rules}."

    def __eq__(self, other):
        if not isinstance(other, JobManager):
            return False
        return self.job_rules == other.job_rules

    def get_job_rules(self):
        return self.job_rules

    def set_job_rules(self, job_rules):
        self.job_rules = job_rules
        return self.job_rules

    def get_job_role(self, job_name: str):
        return self.job_rules.check_job_name_validity(job_name)

    def get_job_role_details(self, job_name: str):
        for job_role in self.job_rules.ValidJobRoles:
            if job_role.get_job_name() == job_name:
                return job_role
        return self.job_rules.ValidJobRoles[0]

    def get_job_role_skills(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_skills()

    def get_job_role_education(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_education()

    def get_job_role_experience(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_experience()

    def get_job_role_motives(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_motives()

    def get_job_role_salary(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_salary()

    def get_job_role_description(self, job_name: str):
        job_role = self.get_job_role_details(job_name)
        return job_role.get_job_description()

    def get_all_job_roles(self):
        return self.job_rules.ValidJobRoles

    def get_all_job_role_names(self):
        job_role_names = []
        for job_role in self.job_rules.ValidJobRoles:
            job_role_names.append(job_role.get_job_name())
        return job_role_names

    def get_all_job_role_skills(self):
        job_role_skills = {}
        for job_role in self.job_rules.ValidJobRoles:
            job_role_skills[job_role.get_job_name()] = job_role.get_job_skills()
