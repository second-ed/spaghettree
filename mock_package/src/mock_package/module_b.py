from typing import Any, Dict, List


def func_e(a: int, b: int) -> int:
    return a + b + func_d()


def func_d():
    return -2


class MockService:
    def fetch_data(self, param: str) -> List[Dict[str, Any]]:
        """
        Mock method to fetch data based on a parameter.

        Args:
            param (str): The parameter to fetch data for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the fetched data.
        """
        return [{"id": 1, "value": "data1"}, {"id": 2, "value": "data2"}]

    def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mock method to process data.

        Args:
            data (List[Dict[str, Any]]): A list of dictionaries containing the data to be processed.

        Returns:
            Dict[str, Any]: A dictionary containing the processed data.
        """
        processed_data = {"summary": "Processed Data", "count": len(data)}
        return processed_data

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Mock method to validate data.

        Args:
            data (List[Dict[str, Any]]): A list of dictionaries containing the data to be validated.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        for item in data:
            if "id" not in item or "value" not in item:
                return False
        return True

    def save_data(self, data: Dict[str, Any]) -> bool:
        """
        Mock method to save data.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to be saved.

        Returns:
            bool: True if the data was saved successfully, False otherwise.
        """
        return True

    def run(self):
        data = self.fetch_data("something")
        if self.validate_data(data):
            processed_data = self.process_data(data)
            self.save_data(processed_data)
