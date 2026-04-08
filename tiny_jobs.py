import json
import logging
import os
import re


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

    def get_job_salary_value(self):
        if isinstance(self.job_salary, (int, float)):
            return float(self.job_salary)
        if isinstance(self.job_salary, str):
            cleaned_salary = re.sub(r"[^0-9.\-]", "", self.job_salary)
            if cleaned_salary:
                return float(cleaned_salary)
        return 0.0

    def get_job_hourly_rate(self, hours_per_day=8, workdays_per_year=260):
        yearly_hours = max(hours_per_day * workdays_per_year, 1)
        return round(self.get_job_salary_value() / yearly_hours, 2)

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
        job_roles_path = os.path.join(os.path.dirname(__file__), "job_roles.json")
        with open(job_roles_path, encoding="utf-8") as job_roles_file:
            job_roles = json.load(job_roles_file)
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

    def find_job_role(self, job_name: str):
        if isinstance(job_name, JobRoles):
            return job_name
        if not isinstance(job_name, str):
            return None
        normalized_job_name = job_name.strip().lower()
        for job_role in self.ValidJobRoles:
            role_name = job_role.get_job_name().strip().lower()
            role_title = job_role.job_title.strip().lower()
            if (
                normalized_job_name == role_name
                or normalized_job_name == role_title
                or normalized_job_name in role_name
                or normalized_job_name in role_title
                or role_name in normalized_job_name
                or role_title in normalized_job_name
            ):
                return job_role
        return None


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
        self.employee = None
        self.job_status = "open"
        self.performance_history = []
        self.last_output = {}

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

    def is_open(self):
        return self.available and self.employee is None

    def assign_employee(self, character):
        self.employee = character
        self.available = False
        self.job_status = "filled"
        return self.employee

    def remove_employee(self):
        previous_employee = self.employee
        self.employee = None
        self.available = True
        self.job_status = "open"
        return previous_employee

    def record_work_outcome(self, outcome):
        self.last_output = outcome
        self.performance_history.append(outcome)
        return self.last_output


class JobManager:
    def __init__(self):
        self.job_rules = JobRules()
        self.active_jobs = {}
        self.character_job_assignments = {}
        self.village_economy = {
            "total_wages_paid": 0.0,
            "resource_value_generated": 0.0,
            "goods_produced": {},
            "services_provided": {},
            "actions_performed": [],
        }

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

    def resolve_job_role(self, job_name: str):
        return self.job_rules.find_job_role(job_name)

    def get_job_role_details(self, job_name: str):
        job_role = self.resolve_job_role(job_name)
        if job_role is not None:
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
        return job_role_skills

    def _get_character_identifier(self, character):
        return getattr(character, "uuid", None) or getattr(character, "name", None) or id(character)

    def _get_character_attribute(self, character, attribute_name, default=None):
        getter_name = f"get_{attribute_name}"
        if hasattr(character, getter_name) and callable(getattr(character, getter_name)):
            try:
                return getattr(character, getter_name)()
            except TypeError:
                pass
        return getattr(character, attribute_name, default)

    def _set_character_attribute(self, character, attribute_name, value):
        setter_name = f"set_{attribute_name}"
        if hasattr(character, setter_name) and callable(getattr(character, setter_name)):
            try:
                getattr(character, setter_name)(value)
                return value
            except TypeError:
                pass
        setattr(character, attribute_name, value)
        return value

    def _set_character_job(self, character, job):
        setattr(character, "job", job)
        if job not in [None, "unemployed"] and hasattr(character, "set_job_role"):
            try:
                character.set_job_role(job)
            except Exception:
                pass
        return getattr(character, "job", job)

    def _normalize_skill_name(self, skill_name):
        return re.sub(r"[^a-z0-9]+", " ", str(skill_name).lower()).strip()

    def _get_character_skills(self, character):
        raw_skills = self._get_character_attribute(character, "skills", {})
        if isinstance(raw_skills, dict):
            return {
                self._normalize_skill_name(skill_name): max(float(level), 0.0)
                for skill_name, level in raw_skills.items()
            }
        if isinstance(raw_skills, (list, tuple, set)):
            return {
                self._normalize_skill_name(skill_name): 1.0
                for skill_name in raw_skills
            }
        return {}

    def _get_job_experience_hours(self, character, job_name):
        experience = self._get_character_attribute(character, "job_experience", {})
        if isinstance(experience, dict):
            return float(experience.get(job_name, 0.0))
        if isinstance(experience, (int, float)):
            return float(experience)
        return 0.0

    def _add_job_experience_hours(self, character, job_name, hours_worked):
        experience = self._get_character_attribute(character, "job_experience", {})
        if not isinstance(experience, dict):
            experience = {}
        experience[job_name] = float(experience.get(job_name, 0.0)) + float(hours_worked)
        self._set_character_attribute(character, "job_experience", experience)
        return experience[job_name]

    def _clamp(self, value, minimum=None, maximum=None):
        if minimum is not None:
            value = max(value, minimum)
        if maximum is not None:
            value = min(value, maximum)
        return value

    def _resolve_job_instance(self, job_or_name):
        if isinstance(job_or_name, Job):
            return job_or_name
        job_role = self.resolve_job_role(job_or_name)
        if job_role is None:
            return None
        return Job(
            job_name=job_role.get_job_name(),
            job_title=job_role.job_title,
            job_description=job_role.get_job_description(),
            job_salary=job_role.get_job_salary(),
            job_skills=job_role.get_job_skills(),
            job_education=job_role.get_job_education(),
            req_job_experience=job_role.get_job_experience(),
            job_motives=job_role.get_job_motives(),
            location=job_role.location,
        )

    def get_character_job(self, character):
        current_job = self.character_job_assignments.get(self._get_character_identifier(character))
        if current_job is not None:
            return current_job
        current_job = self._get_character_attribute(character, "job", None)
        if isinstance(current_job, Job):
            return current_job
        if isinstance(current_job, JobRoles):
            return self._resolve_job_instance(current_job)
        if isinstance(current_job, str) and current_job != "unemployed":
            return self._resolve_job_instance(current_job)
        return None

    def evaluate_job_fit(self, character, job_or_name):
        job_role = self.resolve_job_role(job_or_name)
        if job_role is None:
            return {
                "qualified": False,
                "score": 0.0,
                "matching_skills": [],
                "missing_skills": [],
                "job_role": None,
            }

        character_skills = self._get_character_skills(character)
        job_skills = [
            self._normalize_skill_name(skill_name)
            for skill_name in job_role.get_job_skills()
        ]
        matching_skills = [
            skill_name for skill_name in job_skills if skill_name in character_skills
        ]
        missing_skills = [
            skill_name for skill_name in job_skills if skill_name not in character_skills
        ]
        if not job_skills:
            fit_score = 75.0
        elif not character_skills:
            fit_score = 60.0
        else:
            fit_score = 40.0 + (len(matching_skills) / len(job_skills)) * 60.0

        qualified = not job_skills or not character_skills or len(matching_skills) >= max(1, len(job_skills) // 3)
        return {
            "qualified": qualified,
            "score": round(fit_score, 2),
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "job_role": job_role,
        }

    def character_qualifies_for_job(self, character, job_or_name):
        return self.evaluate_job_fit(character, job_or_name)["qualified"]

    def assign_character_to_job(self, character, job_or_name):
        job = self._resolve_job_instance(job_or_name)
        if job is None:
            return None

        current_job = self.get_character_job(character)
        if current_job is not None and current_job.get_job_name() != job.get_job_name():
            self.leave_job(character)

        job.assign_employee(character)
        self._set_character_job(character, job)

        character_id = self._get_character_identifier(character)
        self.active_jobs[character_id] = job
        self.character_job_assignments[character_id] = job
        return job

    def apply_for_job(self, character, job_or_name, auto_accept=True):
        job_fit = self.evaluate_job_fit(character, job_or_name)
        if not job_fit["qualified"]:
            return {
                "assigned": False,
                "qualified": False,
                "job": None,
                "score": job_fit["score"],
                "matching_skills": job_fit["matching_skills"],
                "missing_skills": job_fit["missing_skills"],
            }

        job = self._resolve_job_instance(job_fit["job_role"])
        if job is None:
            return {
                "assigned": False,
                "qualified": False,
                "job": None,
                "score": 0.0,
                "matching_skills": [],
                "missing_skills": [],
            }

        if auto_accept:
            job = self.assign_character_to_job(character, job)
        return {
            "assigned": auto_accept and job is not None,
            "qualified": True,
            "job": job,
            "score": job_fit["score"],
            "matching_skills": job_fit["matching_skills"],
            "missing_skills": job_fit["missing_skills"],
        }

    def leave_job(self, character):
        character_id = self._get_character_identifier(character)
        current_job = self.get_character_job(character)
        if current_job is None:
            return None

        if isinstance(current_job, Job):
            current_job.remove_employee()
        self._set_character_job(character, "unemployed")
        if hasattr(character, "job_role"):
            try:
                setattr(character, "job_role", None)
            except Exception:
                pass
        self.active_jobs.pop(character_id, None)
        self.character_job_assignments.pop(character_id, None)
        return current_job

    def get_job_actions(self, character_or_job):
        job = character_or_job
        if not isinstance(character_or_job, JobRoles):
            job = self.get_character_job(character_or_job)
        if job is None:
            return []

        title = str(job.job_title or job.get_job_name())
        normalized_title = self._normalize_skill_name(title)
        base_value = max(job.get_job_hourly_rate() * 8, 1)
        action_templates = [
            {
                "keywords": ["farm", "gardener", "agric"],
                "name": "WorkAtFarm",
                "resource_output": {"food": 6},
                "service_output": {},
                "community_impact": 1,
            },
            {
                "keywords": ["blacksmith", "smith", "metal", "mechanic"],
                "name": "CraftToolAtBlacksmith",
                "resource_output": {"tools": 3},
                "service_output": {},
                "community_impact": 1,
            },
            {
                "keywords": ["teacher", "professor", "tutor"],
                "name": "TeachVillagers",
                "resource_output": {},
                "service_output": {"education": 4},
                "community_impact": 2,
            },
            {
                "keywords": ["doctor", "nurse", "healer", "therap"],
                "name": "ProvideMedicalCare",
                "resource_output": {},
                "service_output": {"healthcare": 4},
                "community_impact": 2,
            },
            {
                "keywords": ["chef", "cook", "baker", "barista"],
                "name": "PrepareMeals",
                "resource_output": {"food": 4},
                "service_output": {"hospitality": 2},
                "community_impact": 1,
            },
            {
                "keywords": ["builder", "architect", "engineer", "construction"],
                "name": "BuildVillageInfrastructure",
                "resource_output": {"infrastructure": 2},
                "service_output": {},
                "community_impact": 2,
            },
        ]

        selected_template = None
        for template in action_templates:
            if any(keyword in normalized_title for keyword in template["keywords"]):
                selected_template = template
                break

        if selected_template is None:
            generated_name = re.sub(r"[^A-Za-z0-9]+", "", title.title()) or "CurrentJob"
            selected_template = {
                "name": f"Perform{generated_name}",
                "resource_output": {},
                "service_output": {"labor": 3},
                "community_impact": 1,
            }

        primary_action = {
            "name": selected_template["name"],
            "description": f"Perform a productive shift as {title}.",
            "time_cost": 8,
            "energy_cost": 2,
            "performance_gain": 4,
            "resource_output": selected_template["resource_output"],
            "service_output": selected_template["service_output"],
            "community_impact": selected_template["community_impact"],
            "economic_value": round(base_value, 2),
            "skill_focus": list(job.get_job_skills()),
        }
        skill_action = {
            "name": f"Improve{re.sub(r'[^A-Za-z0-9]+', '', title.title()) or 'Job'}Skills",
            "description": f"Study and practice skills related to {title}.",
            "time_cost": 2,
            "energy_cost": 1,
            "performance_gain": 2,
            "resource_output": {},
            "service_output": {},
            "community_impact": 0,
            "economic_value": 0.0,
            "skill_focus": list(job.get_job_skills()),
        }
        return [primary_action, skill_action]

    def _develop_job_skills(self, character, skill_names, growth_amount=1.0):
        if not skill_names:
            return {}
        raw_skills = self._get_character_attribute(character, "skills", {})
        if not isinstance(raw_skills, dict):
            raw_skills = {}
        updated_skills = {}
        for skill_name in skill_names:
            current_level = float(raw_skills.get(skill_name, 0.0))
            new_level = round(current_level + growth_amount, 2)
            raw_skills[skill_name] = new_level
            updated_skills[skill_name] = new_level
        self._set_character_attribute(character, "skills", raw_skills)
        return updated_skills

    def perform_job(self, character, job=None, action_name=None, hours_worked=None):
        active_job = self.get_character_job(character)
        if active_job is None and job is not None:
            active_job = self.assign_character_to_job(character, job)
        if active_job is None:
            return {
                "success": False,
                "reason": "Character is unemployed.",
            }

        available_actions = self.get_job_actions(active_job)
        if not available_actions:
            return {
                "success": False,
                "reason": "No job actions are available.",
            }

        selected_action = available_actions[0]
        if action_name:
            for action in available_actions:
                if action["name"] == action_name:
                    selected_action = action
                    break

        if hours_worked is None:
            hours_worked = selected_action["time_cost"]

        current_performance = float(self._get_character_attribute(character, "job_performance", 20.0))
        current_energy = float(self._get_character_attribute(character, "energy", 8.0))
        current_hunger = float(self._get_character_attribute(character, "hunger_level", 2.0))
        current_wealth = float(self._get_character_attribute(character, "wealth_money", 0.0))
        current_material_goods = float(self._get_character_attribute(character, "material_goods", 0.0))
        current_community = float(self._get_character_attribute(character, "community", 5.0))

        skill_fit = self.evaluate_job_fit(character, active_job)
        skill_bonus = min(len(skill_fit["matching_skills"]), 3)
        hunger_penalty = max(current_hunger - 4.0, 0.0) * 0.5
        energy_penalty = max(4.0 - current_energy, 0.0) * 0.5
        performance_change = round(
            selected_action["performance_gain"] + skill_bonus - hunger_penalty - energy_penalty,
            2,
        )
        new_performance = self._clamp(current_performance + performance_change, 0.0, 100.0)

        productivity_multiplier = max(0.25, 0.5 + new_performance / 100.0)
        income = round(active_job.get_job_hourly_rate() * float(hours_worked) * productivity_multiplier, 2)
        goods_created = {
            item_name: round(quantity * productivity_multiplier, 2)
            for item_name, quantity in selected_action["resource_output"].items()
        }
        services_created = {
            service_name: round(quantity * productivity_multiplier, 2)
            for service_name, quantity in selected_action["service_output"].items()
        }

        new_energy = self._clamp(current_energy - selected_action["energy_cost"], 0.0, 10.0)
        new_hunger = self._clamp(current_hunger + max(hours_worked / 4.0, 1.0), 0.0, 10.0)
        new_wealth = round(current_wealth + income, 2)
        new_material_goods = round(
            current_material_goods + sum(goods_created.values()),
            2,
        )
        new_community = self._clamp(
            current_community + selected_action["community_impact"],
            0.0,
            10.0,
        )

        self._set_character_attribute(character, "job_performance", new_performance)
        self._set_character_attribute(character, "energy", new_energy)
        self._set_character_attribute(character, "hunger_level", new_hunger)
        self._set_character_attribute(character, "wealth_money", new_wealth)
        self._set_character_attribute(character, "material_goods", new_material_goods)
        self._set_character_attribute(character, "community", new_community)
        skill_updates = self._develop_job_skills(
            character,
            selected_action["skill_focus"],
            growth_amount=max(hours_worked / 8.0, 0.5),
        )
        total_experience = self._add_job_experience_hours(
            character,
            active_job.get_job_name(),
            hours_worked,
        )

        self.village_economy["total_wages_paid"] = round(
            self.village_economy["total_wages_paid"] + income,
            2,
        )
        produced_value = round(
            income + sum(goods_created.values()) + sum(services_created.values()),
            2,
        )
        self.village_economy["resource_value_generated"] = round(
            self.village_economy["resource_value_generated"] + produced_value,
            2,
        )
        for item_name, quantity in goods_created.items():
            self.village_economy["goods_produced"][item_name] = round(
                self.village_economy["goods_produced"].get(item_name, 0.0) + quantity,
                2,
            )
        for service_name, quantity in services_created.items():
            self.village_economy["services_provided"][service_name] = round(
                self.village_economy["services_provided"].get(service_name, 0.0) + quantity,
                2,
            )

        outcome = {
            "success": True,
            "job_name": active_job.get_job_name(),
            "job_title": active_job.job_title,
            "action": selected_action["name"],
            "income": income,
            "energy_spent": selected_action["energy_cost"],
            "hours_worked": hours_worked,
            "performance_change": round(new_performance - current_performance, 2),
            "new_job_performance": new_performance,
            "resource_output": goods_created,
            "service_output": services_created,
            "skill_updates": skill_updates,
            "job_experience_hours": total_experience,
            "generated_value": produced_value,
            "promotion_available": self.find_promotion_opportunity(character) is not None,
        }
        active_job.record_work_outcome(outcome)
        self.village_economy["actions_performed"].append(outcome)
        return outcome

    def _job_family_tokens(self, job_role):
        title = job_role.job_title or job_role.get_job_name()
        tokens = {
            token
            for token in self._normalize_skill_name(title).split()
            if token
            and token
            not in {"junior", "senior", "lead", "assistant", "associate", "manager"}
        }
        return tokens

    def find_promotion_opportunity(self, character, current_job=None):
        active_job = self.get_character_job(character)
        if current_job is not None:
            active_job = self._resolve_job_instance(current_job)
        if active_job is None:
            return None

        current_role = self.resolve_job_role(active_job)
        if current_role is None:
            return None

        current_salary = current_role.get_job_salary_value()
        current_performance = float(self._get_character_attribute(character, "job_performance", 0.0))
        total_experience = self._get_job_experience_hours(character, current_role.get_job_name())
        if current_performance < 70.0 or total_experience < 40.0:
            return None

        current_family_tokens = self._job_family_tokens(current_role)
        current_skill_tokens = {
            self._normalize_skill_name(skill_name)
            for skill_name in current_role.get_job_skills()
        }
        candidates = []
        for candidate in self.job_rules.ValidJobRoles:
            if candidate.get_job_name() == current_role.get_job_name():
                continue
            candidate_salary = candidate.get_job_salary_value()
            if candidate_salary <= current_salary:
                continue
            candidate_family_tokens = self._job_family_tokens(candidate)
            candidate_skill_tokens = {
                self._normalize_skill_name(skill_name)
                for skill_name in candidate.get_job_skills()
            }
            family_overlap = len(current_family_tokens & candidate_family_tokens)
            skill_overlap = len(current_skill_tokens & candidate_skill_tokens)
            if family_overlap == 0 and skill_overlap == 0:
                continue
            candidates.append(
                (
                    family_overlap,
                    skill_overlap,
                    -(candidate_salary - current_salary),
                    candidate,
                )
            )

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][-1]

    def progress_career(self, character, auto_promote=True):
        current_job = self.get_character_job(character)
        if current_job is None:
            return {
                "promoted": False,
                "reason": "Character has no active job.",
            }

        promotion = self.find_promotion_opportunity(character, current_job=current_job)
        if promotion is None:
            skill_growth = self._develop_job_skills(
                character,
                current_job.get_job_skills(),
                growth_amount=0.5,
            )
            return {
                "promoted": False,
                "current_job": current_job.get_job_name(),
                "skill_growth": skill_growth,
            }

        if not auto_promote:
            return {
                "promoted": False,
                "promotion_available": True,
                "current_job": current_job.get_job_name(),
                "next_role": promotion.get_job_name(),
            }

        previous_job_name = current_job.get_job_name()
        promoted_job = self.assign_character_to_job(character, promotion)
        salary_increase = round(
            promoted_job.get_job_salary_value() - current_job.get_job_salary_value(),
            2,
        )
        return {
            "promoted": True,
            "previous_job": previous_job_name,
            "new_job": promoted_job.get_job_name(),
            "salary_increase": salary_increase,
        }

    def get_village_economy_summary(self):
        return {
            "total_wages_paid": self.village_economy["total_wages_paid"],
            "resource_value_generated": self.village_economy["resource_value_generated"],
            "goods_produced": dict(self.village_economy["goods_produced"]),
            "services_provided": dict(self.village_economy["services_provided"]),
            "actions_performed": list(self.village_economy["actions_performed"]),
        }
