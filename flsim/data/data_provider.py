#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, List, Optional

from flsim.interfaces.model import IFLModel


class IFLUserData(ABC):
    """
    Wraps data for a single user

    IFLUserData is responsible for
        1. Keeping track of the number of examples for a particular user
        2. Keeping track of the number of batches for a particular user
        3. Providing an iterator over all the user batches
    """

    def num_total_examples(self) -> int:
        """
        Returns the number of examples
        """
        return self.num_train_examples() + self.num_eval_examples()

    def num_total_batches(self) -> int:
        """
        Returns the number of batches
        """
        return self.num_train_batches() + self.num_eval_batches()

    @abstractmethod
    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """

    @abstractmethod
    def num_eval_examples(self) -> int:
        """
        Returns the number of eval examples
        """

    @abstractmethod
    def num_train_batches(self) -> int:
        """
        Returns the number of training batches
        """

    @abstractmethod
    def num_eval_batches(self) -> int:
        """
        Returns the number of eval batches
        """

    @abstractmethod
    def train_data(self) -> Iterator[Any]:
        """
        Returns the training batches
        """

    @abstractmethod
    def eval_data(self) -> Iterator[Any]:
        """
        Returns the eval batches
        """


class IFLDataProvider(ABC):
    """
    Provides data to the trainer

    IFLDataProvider is responsible for
        1. Enforcing a uniform interface that trainer expects
        2. Transforming data into what IFLModel.fl_forward() is going to consume
        3. Keeping track of the sharded client data
    """

    @abstractmethod
    def train_user_ids(self) -> List[int]:
        """
        Returns a list of user ids in the data set
        """

    @abstractmethod
    def num_train_users(self) -> int:
        """
        Returns the number of users in train set
        """

    @abstractmethod
    def get_train_user(self, user_index: int) -> IFLUserData:
        """
        Returns train user from user_index
        """

    @abstractmethod
    def train_users(self) -> Iterable[IFLUserData]:
        """
        Returns training users iterable
        """

    @abstractmethod
    def eval_users(self) -> Iterable[IFLUserData]:
        """
        Returns evaluation users iterable
        """

    @abstractmethod
    def test_users(self) -> Iterable[IFLUserData]:
        """
        Returns test users iterable
        """


class FLUserDataFromList(IFLUserData):
    """
    Util class to create an IFLUserData from a list of user batches
    """

    def __init__(
        self, data: Iterable, model: IFLModel, eval_batches: Optional[Iterable] = None
    ):
        self.data = data
        self._num_examples: int = 0
        self._num_batches: int = 0
        self.model = model
        self.training_batches = []
        self.eval_batches = eval_batches if eval_batches is not None else []
        self._num_eval_batches: int = 0
        self._num_eval_examples: int = 0

        for batch in self.data:
            training_batch = self.model.fl_create_training_batch(batch=batch)
            self.training_batches.append(training_batch)
            self._num_examples += model.get_num_examples(training_batch)
            self._num_batches += 1

        for batch in self.eval_batches:
            eval_batch = self.model.fl_create_training_batch(batch=batch)
            self._num_eval_examples += model.get_num_examples(eval_batch)
            self._num_eval_batches += 1

    def train_data(self):
        for batch in self.training_batches:
            yield batch

    def eval_data(self):
        for batch in self.eval_batches:
            yield self.model.fl_create_training_batch(batch=batch)

    def num_batches(self):
        return self._num_batches

    def num_train_examples(self):
        return self._num_examples

    def num_eval_batches(self):
        return self._num_eval_batches

    def num_train_batches(self):
        return self._num_batches

    def num_eval_examples(self):
        return self._num_eval_examples

class UserDataWithTier(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split: float = 0.0):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split

        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        self._user_tier = int(next(user_data["tier"])[0])
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += UserDataWithTier.get_num_examples(labels)
                self._eval_batches.append(UserDataWithTier.fl_training_batch(features, labels))
            else:
                self._num_train_batches += 1
                self._num_train_examples += UserDataWithTier.get_num_examples(labels)
                self._train_batches.append(UserDataWithTier.fl_training_batch(features, labels))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches
    
    def user_tier(self):
        return self._user_tier

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            yield batch

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            yield batch

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float]
    ) -> Dict[str, torch.Tensor]:
        return {"features": torch.stack(features), "labels": torch.Tensor(labels)}



class FLDataProviderFromList(IFLDataProvider):
    """Utility class to help ease the transition to IFLDataProvider

    Args:
        train_user_list: (Iterable[Iterable[Any]]): train data
        eval_user_list: (Iterable[Iterable[Any]]): eval data
        test_user_list (Iterable[Iterable[Any]]): test data
        model: (IFLModel): the IFLModel to create training batch for
    """

    def __init__(
        self,
        train_user_list: Iterable[Iterable[Any]],
        eval_user_list: Iterable[Iterable[Any]],
        test_user_list: Iterable[Iterable[Any]],
        model: IFLModel,
    ):
        self.train_user_list = train_user_list
        self.eval_user_list = eval_user_list
        self.test_user_list = test_user_list
        self.model = model
        self._train_users = {
            user_id: FLUserDataFromList(
                data=user_data, eval_batches=user_data, model=model
            )
            for user_id, user_data in enumerate(train_user_list)
        }
        self._eval_users = {
            user_id: FLUserDataFromList(data=[], eval_batches=user_data, model=model)
            for user_id, user_data in enumerate(eval_user_list)
        }
        self._test_users = {
            user_id: FLUserDataFromList(data=[], eval_batches=user_data, model=model)
            for user_id, user_data in enumerate(test_user_list)
        }

    def train_user_ids(self):
        """List of all train user IDs."""
        return list(self._train_users.keys())

    def num_train_users(self):
        """Number of train users."""
        return len(self.train_user_ids())

    def get_train_user(self, user_index: int):
        """Returns a train user given user index."""
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(f"Index {user_index} not in {self.train_user_ids()}")

    def train_users(self):
        """Returns a list of all train users."""
        return list(self._train_users.values())

    def eval_users(self):
        """Returns a list of all eval users."""
        return list(self._eval_users.values())

    def test_users(self):
        """Returns a list of test users."""
        return list(self._test_users.values())

class DataProviderWithTier(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: UserDataWithTier(user_data, eval_split=eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }