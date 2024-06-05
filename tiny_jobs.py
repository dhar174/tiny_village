class JobRoles:
    def __init__(
        self,
        job_name,
        job_description,
        job_salary,
        job_skills,
        job_education,
        job_experience,
        job_motives,
    ):
        self.job_name = self.set_job_name(job_name)
        self.job_description = self.set_job_description(job_description)
        self.job_salary = self.set_job_salary(job_salary)
        self.job_skills = self.set_job_skills(job_skills)
        self.job_education = self.set_job_education(job_education)
        self.job_experience = self.set_job_experience(job_experience)
        self.job_motives = self.set_job_motives(job_motives)

    def __repr__(self):
        return f"JobRoles({self.job_name}, {self.job_description}, {self.job_salary}, {self.job_skills}, {self.job_education}, {self.job_experience}, {self.job_motives})"

    def __str__(self):
        return f"JobRoles with name {self.job_name}, description {self.job_description}, salary {self.job_salary}, skills {self.job_skills}, education {self.job_education}, experience {self.job_experience}, motives {self.job_motives}."

    def __eq__(self, other):
        return (
            self.job_name == other.job_name
            and self.job_description == other.job_description
            and self.job_salary == other.job_salary
            and self.job_skills == other.job_skills
            and self.job_education == other.job_education
            and self.job_experience == other.job_experience
            and self.job_motives == other.job_motives
        )

    def __hash__(self):
        return hash(
            (
                self.job_name,
                self.job_description,
                self.job_salary,
                self.job_skills,
                self.job_education,
                self.job_experience,
                self.job_motives,
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
        return self.job_experience

    def set_job_experience(self, job_experience):
        self.job_experience = job_experience
        return self.job_experience

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
            "experience": self.job_experience,
            "motives": self.job_motives,
        }


class JobRules:
    def __init__(self):

        self.ValidJobRoles = [
            JobRoles("unemployed", "no job", 0, [], [], [], []),
            JobRoles(
                "cashier",
                "works at a grocery store",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "chef",
                "works at a restaurant",
                10,
                [
                    "cooking",
                    "customer service",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "janitor",
                "cleans buildings",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "teacher",
                "teaches students",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "police officer",
                "enforces the law",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "doctor",
                "treats patients",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "engineer",
                "builds things",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "farmer",
                "grows crops",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "firefighter",
                "fights fires",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "lawyer",
                "represents clients in court",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
            JobRoles(
                "mechanic",
                "repairs vehicles",
                10,
                [
                    "customer service",
                    "mathematics",
                    "communication",
                    "time management",
                    "organization",
                ],
                ["high school diploma"],
                ["none"],
                ["wealth", "happiness", "job performance"],
            ),
        ]

    def __repr__(self):
        return f"JobRules({self.ValidJobRoles})"

    def __str__(self):
        return f"JobRules with valid job roles {self.ValidJobRoles}."

    def __eq__(self, other):
        return self.ValidJobRoles == other.ValidJobRoles

    def check_job_role_validity(self, job_role: JobRoles):
        if job_role in self.ValidJobRoles:
            return True
        else:
            return False

    def check_job_name_validity(self, job_name: str):
        for job_role in self.ValidJobRoles:
            if job_role.get_job_name() == job_name:
                return job_name
        return "unemployed"


# Job class is a subclass of JobRoles and inherits from it
class Job(JobRoles):
    def __init__(
        self,
        job_name,
        job_description,
        job_salary,
        job_skills,
        job_education,
        job_experience,
        job_motives,
    ):
        super().__init__(
            job_name,
            job_description,
            job_salary,
            job_skills,
            job_education,
            job_experience,
            job_motives,
        )
        # Warning: Name MUST be unique! Check for duplicates before setting.

        self.job_name = job_name
        self.job_description = job_description
        self.job_salary = job_salary
        self.job_skills = job_skills
        self.job_education = job_education
        self.job_experience = job_experience
        self.job_motives = job_motives

    def __repr__(self):
        return f"Job({self.job_name}, {self.job_description}, {self.job_salary}, {self.job_skills}, {self.job_education}, {self.job_experience}, {self.job_motives})"

    def __str__(self):
        return f"Job with name {self.job_name}, description {self.job_description}, salary {self.job_salary}, skills {self.job_skills}, education {self.job_education}, experience {self.job_experience}, motives {self.job_motives}."

    def __eq__(self, other):
        return (
            self.job_name == other.job_name
            and self.job_description == other.job_description
            and self.job_salary == other.job_salary
            and self.job_skills == other.job_skills
            and self.job_education == other.job_education
            and self.job_experience == other.job_experience
            and self.job_motives == other.job_motives
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


class JobManager:
    def __init__(self):
        self.job_rules = JobRules()

    def __repr__(self):
        return f"JobManager({self.job_rules})"

    def __str__(self):
        return f"JobManager with job rules {self.job_rules}."

    def __eq__(self, other):
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
