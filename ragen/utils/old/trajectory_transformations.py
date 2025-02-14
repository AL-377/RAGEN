class TrajectoryTransformation:
    """
    Base class for all trajectory transformation strategies.
    Subclasses must implement `generate_pairs` and `create_prompt`.
    """

    @staticmethod
    def generate_pairs(steps):
        raise NotImplementedError("Subclasses must implement `generate_pairs`.")

    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False):
        raise NotImplementedError("Subclasses must implement `create_prompt`.")

    @classmethod
    def in_context_examples(cls):
        raise NotImplementedError("Subclasses must implement `in_context_examples`.")
    
    @classmethod
    def task_intro(cls):
        intro = (
            "You are a Sokoban solver.\n"
            "\n"
            "Sokoban Quick Guide\n" + "Goal: Push all boxes (X) onto targets (O).\n"
            "\n"
            "Symbols:\n" + "# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target\n"
            "\n"
            "Rules:\n" + "1. Push boxes (can't pull).\n" + "2. Avoid walls (#).\n"
            "\n"
            "Answers:\n" + "<answer> 1 (Up) </answer> | <answer> 2 (Down) </answer> | <answer> 3 (Left) </answer> | <answer> 4 (Right) </answer>\n"
            "\n"
            "Rewards:\n" + "Move: -0.1\n" + "Box on target: +1.0\n" + "All boxes placed: +10.0\n"
            "\n"
        )
        return intro
    
    def transform(self, steps):
        pairs = self.generate_pairs(steps) # steps has all traj information, please refer to utils/dataset.py for more details
        output_list = []
        for pair in pairs:
            prompt, prediction = self.create_prompt(pair["condition"], pair["prediction"])
            output_list.append({"prompt": prompt, "prediction": prediction})
        return output_list


class TaskPlanning(TrajectoryTransformation):
    @staticmethod
    def generate_pairs(steps):
        return [
            {
                "condition": {
                    "all-observation": steps[i]["all-observation"],
                    "all-observation-list": steps[i]["all-observation-list"]
                },
                "prediction": [s[1] for s in steps[i]['best_future_trajectory']]
            }
            for i in range(len(steps))
        ]

    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False, task_intro=True):
        examples = cls.in_context_examples() if context_example else ""
        task_intro = cls.task_intro() if task_intro else ""
        return (
            f"{task_intro}\n{examples}[Cumulative Observations]:\n{condition['all-observation']}\nGenerate a sequence of actions:",
            " ".join(map(str, prediction))
        )

    @classmethod
    def in_context_examples(cls):
        return (
            "[Examples]\n"
            "1. Observations:\n# # # # #\n# _ P _ #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            "Generate a sequence of actions:\n3 2\n\n"
            "2. Observations:\n# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n"
            "Generate a sequence of actions:\n4 1 4 2\n\n"
            "3. Observations:\n# # # # #\n# P _ _ #\n# _ X _ #\n# _ _ O #\n# # # # #\n"
            "Generate a sequence of actions:\n2 4 1 4 2\n\n"
            "4. Observations:\n# # # # #\n# _ _ _ #\n# _ _ _ #\n# O X P #\n# # # # #\n"
            "Generate a sequence of actions:\n3\n\n"
        )


class DecisionMaking(TrajectoryTransformation):
    @staticmethod
    def generate_pairs(steps):
        return [
            {
                "condition": {
                    "all-observation": steps[i]["all-observation"],
                    "all-observation-list": steps[i]["all-observation-list"]
                },
                "prediction": [steps[i]["action"]]
            }
            for i in range(len(steps))
        ]

    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False, task_intro=True):
        examples = cls.in_context_examples() if context_example else ""
        task_intro = cls.task_intro() if task_intro else ""
        return (
            f"{task_intro}\n{examples}[Cumulative Observations]:\n{condition['all-observation']}\nDecide the next action:",
            str(prediction[0])
        )

    @classmethod
    def in_context_examples(cls):
        return (
            "[Examples]\n"
            "1. Observations:\n# # # # #\n# _ P _ #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            "Decide the next action:\nAction: 1\n\n"
            "2. Observations:\n# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n"
            "Decide the next action:\nAction: 4\n\n"
            "3. Observations:\n# # # # #\n# P _ _ #\n# _ _ _ #\n# _ X O #\n# # # # #\n"
            "Decide the next action:\nAction: 2\n\n"
            "4. Observations:\n# # # # #\n# _ _ P #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            "Decide the next action:\nAction: 3\n\n"
        )


class ForwardDynamics(TrajectoryTransformation):
    @staticmethod
    def generate_pairs(steps):
        return [
            {
                "condition": {
                    "all-observation": steps[i]["all-observation"],
                    "action": steps[i]["action"]
                },
                "prediction": [steps[i]["next_observation"]]
            }
            for i in range(len(steps))
        ]

    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False, no_acc_obs=True, task_intro=True):
        examples = cls.in_context_examples() if context_example else ""
        task_intro = cls.task_intro() if task_intro else ""
        if no_acc_obs:
            def _diff_observation(obs_ori, obs_new):
                obs_ori = obs_ori.split('\n')
                obs_new = obs_new.split('\n')
                diff = []
                for i in range(len(obs_ori)):
                    for j in range(len(obs_ori[i])):
                        if obs_ori[i][j] != obs_new[i][j]:
                            diff.append((i, j, obs_ori[i][j], obs_new[i][j]))
                return diff
            # # use case:
            # obs_ori = "# # # # #\n# _ P _ #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            # obs_new = "# # # # #\n# _ _ _ #\n# X P _ #\n# O _ _ #\n# # # # #\n"
            # diff = _diff_observation(obs_ori, obs_new)
            # print(obs_ori)
            # print(obs_new)
            # print(diff)
            cur_observation = condition['all-observation'].split('\n\n')[-1]
            diff = _diff_observation(cur_observation, prediction[0])
            diff_str = ""
            for i in range(len(diff)):
                diff_str += f"({diff[i][0]}, {diff[i][1]}): {diff[i][2]} -> {diff[i][3]}\n"

            return (
                f"{task_intro}\n{examples}[Observation]:\n{cur_observation}\nAction Taken: {condition['action']}\nPredict the next observation:",
                diff_str
            )
        else:
            return (
                f"{task_intro}\n{examples}[Cumulative Observations]:\n{condition['all-observation']}\nAction Taken: {condition['action']}\nPredict the next observation:",
                prediction[0]
            )

    @classmethod
    def in_context_examples(cls):
        return (
            "[Examples]\n"
            "1. Observations:\n# # # # #\n# _ P _ #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            "Action Taken: 2\nPredict the next observation:\n"
            "# # # # #\n# _ _ _ #\n# X P _ #\n# O _ _ #\n# # # # #\n\n"
            "2. Observations:\n# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n"
            "Action Taken: 3\nPredict the next observation:\n"
            "# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n\n"
            "3. Observations:\n# # # # #\n# P O _ #\n# _ X _ #\n# _ _ _ #\n# # # # #\n"
            "Action Taken: 4\nPredict the next observation:\n"
            "# # # # #\n# _ S _ #\n# _ X _ #\n# _ _ _ #\n# # # # #\n\n"
            "4. Observations:\n# # # # #\n# O _ _ #\n# X _ _ #\n# P _ _ #\n# # # # #\n"
            "Action Taken: 1\nPredict the next observation:\n"
            "# # # # #\n# √ _ _ #\n# P _ _ #\n# _ _ _ #\n# # # # #\n\n"
        )


class InverseDynamics(TrajectoryTransformation):
    @staticmethod
    def generate_pairs(steps):
        return [
            {
                "condition": {
                    "all-observation": steps[i]["all-observation"],
                    "next-observation": steps[i]["next_observation"]
                },
                "prediction": [steps[i]["action"]]
            }
            for i in range(len(steps))
        ]

    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False, task_intro=True):
        examples = cls.in_context_examples() if context_example else ""
        task_intro = cls.task_intro() if task_intro else ""
        return (
            f"{task_intro}\n{examples}[Cumulative Observations]:\n{condition['all-observation']}\nNext Observation:\n{condition['next-observation']}\nWhat action caused this transition?",
            str(prediction[0])
        )

    @classmethod
    def in_context_examples(cls):
        return (
            "[Examples]\n"
            "1. Observations:\n# # # # #\n# _ P _ #\n# X _ _ #\n# O _ _ #\n# # # # #\n"
            "Next Observation:\n# # # # #\n# _ _ _ #\n# X P _ #\n# O _ _ #\n# # # # #\n"
            "What action caused this transition?\nAction: 2\n\n"
            "2. Observations:\n# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n"
            "Next Observation:\n# # # # #\n# _ _ _ #\n# P _ X #\n# _ _ O #\n# # # # #\n"
            "What action caused this transition?\nAction: 3\n\n"
            "3. Observations:\n# # # # #\n# P O _ #\n# _ X _ #\n# _ _ _ #\n# # # # #\n"
            "Next Observation:\n# # # # #\n# _ S _ #\n# _ X _ #\n# _ _ _ #\n# # # # #\n"
            "What action caused this transition?\nAction: 4\n\n"
            "4. Observations:\n# # # # #\n# O _ _ #\n# X _ _ #\n# P _ _ #\n# # # # #\n"
            "Next Observation:\n# # # # #\n# √ _ _ #\n# P _ _ #\n# _ _ _ #\n# # # # #\n"
            "What action caused this transition?\nAction: 1\n\n"
        )


class RewardPrediction(TrajectoryTransformation):
    @staticmethod
    def generate_pairs(steps):
        return [
            {
                "condition": {
                    "all-observation": steps[i]["all-observation"],
                    "action": steps[i]["action"]
                },
                "prediction": [{"type": "reward", "content": str(steps[i]["reward"])}]
            }
            for i in range(len(steps))
        ]
        
    @classmethod
    def create_prompt(cls, condition, prediction, context_example=False, task_intro=True):
        examples = cls.in_context_examples() if context_example else ""
        task_intro = cls.task_intro() if task_intro else ""
        return (
            f"{task_intro}\n{examples}[Cumulative Observations]:\n{condition['all-observation']}\nAction Taken: {condition['action']}\nWhat is the reward for this step?",
            prediction[0]["content"]
        )

    @classmethod
    def in_context_examples(cls):
        return (
            "[Examples]\n"
            "1. Observations:\n# # # # #\n# _ _ P #\n# _ _ X #\n# _ _ O #\n# # # # #\n"
            "Action Taken: 1\nWhat is the reward for this step?\nReward: -0.1\n\n"
            "2. Observations:\n# # # # #\n# _ P _ #\n# _ X _ #\n# _ O _ #\n# # # # #\n"
            "Action Taken: 2\nWhat is the reward for this step?\nReward: 10.9\n\n"
            "3. Observations:\n# # # # #\n# O X P #\n# _ _ X #\n# _ _ O #\n# # # # #\n"
            "Action Taken: 3\nWhat is the reward for this step?\nReward: 0.9\n\n"
        )


TRANSFORMATION_REGISTRY = {
    "task_planning": TaskPlanning,
    "decision_making": DecisionMaking,
    "forward_dynamics": ForwardDynamics,
    "inverse_dynamics": InverseDynamics,
    "reward_prediction": RewardPrediction,
}
