from pathlib import Path, PurePath
from .core import cord_support_dir, render_html
import pandas as pd


def load_tasks():
    task_csv = cord_support_dir() / 'TaskDefinitions.csv'
    tasks = pd.read_csv(PurePath(task_csv))
    tasks.SeedQuestion = tasks.SeedQuestion.fillna(tasks.Question)
    return tasks


class Task:

    def __init__(self, topics):
        self.topics = topics.drop(columns=['Task', 'Type']).set_index('TopicNo')

    def __getitem__(self, item):
        return self.topics.loc[item]

    def table_of_contents(self):
        links = self.topics.Question.apply(lambda q: q.replace(' ', '-'))
        topics = self.topics.Question.tolist()

        return render_html('TableOfContents',
                           topics=zip(links, topics))

    def _repr_html_(self):
        return self.topics._repr_html_()


class TaskDefinitions:

    @classmethod
    def load(cls):
        task_df = load_tasks()
        tasks = cls()
        for task_name in task_df.Task.drop_duplicates().to_list():
            setattr(tasks, task_name, Task(task_df[task_df.Task == task_name]))
        return tasks

Tasks = TaskDefinitions.load()