from openweights import OpenWeights

ow = OpenWeights()

model = 'unsloth/llama-3-8b-Instruct'

if __name__ == "__main__":

    # async with ow.api.deploy(model) also works
    with ow.api.deploy(model):            # async with ow.api.deploy(model) also works
        # entering the context manager is equivalent to temp_api = ow.api.deploy(model) ; api.up()
        completion = ow.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "is 9.11 > 9.9?"}]
        )
        print(completion.choices[0].message)       # when this context manager exits, it calls api.down()