from src.utils import load_json


class Attacker():
    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')

    def get_attack(self, target_queries) -> list:
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        adv_text_groups = []
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)
        else:
            raise NotImplementedError
        return adv_text_groups
