import pytest
import pandas as pd
from djhjgfv import children  

class TestTitanicAnalysis:
    
    def test_1(self):
        """Данные всех портов"""
        test_data = pd.DataFrame({
            'Age': [5, 10, 15, 8, 12],
            'Survived': [0, 0, 0, 0, 0],
            'Embarked': ['C', 'Q', 'S', 'C', 'Q'],
            'Pclass': [1, 2, 3, 1, 2]
        })
        
        counts, total, max_ages = children(test_data)
        
        assert counts['C'] == 2
        assert counts['Q'] == 2
        assert counts['S'] == 1
        assert total == 5
        assert max_ages[1] == 8
        assert max_ages[2] == 12
        assert max_ages[3] == 15
    
    def test_2(self):
        """Нет погибших детей"""
        test_data = pd.DataFrame({
            'Age': [5, 10, 15],
            'Survived': [1, 1, 1],  
            'Embarked': ['C', 'Q', 'S'],
            'Pclass': [1, 2, 3]
        })
        
        counts, total, max_ages = children(test_data)
        
        assert counts['C'] == 0
        assert counts['Q'] == 0
        assert counts['S'] == 0
        assert total == 0
        assert max_ages[1] == 0
        assert max_ages[2] == 0
        assert max_ages[3] == 0
    
    
    def test_3(self):
        """Тест правильности вычисления максимального возраста по классам"""
        test_data = pd.DataFrame({
            'Age': [10, 5, 15, 8, 12],
            'Survived': [0, 0, 0, 0, 0],
            'Embarked': ['C', 'Q', 'S', 'C', 'Q'],
            'Pclass': [1, 1, 1, 2, 3]  
        })
        
        counts, total, max_ages = children(test_data)
        
        assert max_ages[1] == 10
        assert max_ages[2] == 8
        assert max_ages[3] == 12