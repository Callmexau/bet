# -*- coding: utf-8 -*-
"""
models/__init__.py - Point d'entrée du package models
Expose les classes et fonctions principales pour simplifier les imports.
"""

from models.elo import EloRating
from models.football import FootballModel
from models.basketball import BasketballModel
from models.calibration import TemperatureCalibrator, PlattCalibrator

__all__ = [
    "EloRating",
    "FootballModel",
    "BasketballModel",
    "TemperatureCalibrator",
    "PlattCalibrator",
]
