from functools import wraps


class ImplementationLeftAsExerciseForTheReaderError(NotImplementedError):
    def __init__(self, func: callable):
        super().__init__(
            f"The Implementation of {func} is left as an Exercise for the Reader or User of this Library. 😂😂😂"
        )


def raise_left_as_exercise_for_reader(func):
    @wraps(func)
    def wrapper(*_, **__):
        raise ImplementationLeftAsExerciseForTheReaderError(func=func)
    return wrapper
