from lm_eval import evaluator
from .lm_eval_adaptor import LMEvalAdaptor

def evaluate(model_name, model, tokenizer, tasks, quant_bitwidth=None, limit=None, batch_size=1, max_length=4096):
    """
    tasks: a str separated by comma, eg. 'wikitext,lambada,hellaswag'
    """
    print(f'* Evaluating on tasks: {tasks}')

    lm_eval_model = LMEvalAdaptor(model_name, model, tokenizer, batch_size, max_length)
    tasks = tasks.split(',')

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        batch_size=batch_size,
        no_cache=True,
        limit=limit,
        num_fewshot=0,
    )
    print(f"* Results of {quant_bitwidth if quant_bitwidth else ''} model {model_name}:")
    print(evaluator.make_table(results))
    return results
