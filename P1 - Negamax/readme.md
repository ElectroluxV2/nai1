# Dodge The Pile

Python game vs AI, created for NAI labs.    
Created by: Mateusz Budzisz && Aleksander Guzik



# Dodge The Pile Game

"Dodge The Pile" is a two-player, turn-based game where players take alternating turns manipulating a shared score. The objective of the game is to strategically add or subtract numbers from the score in a way that avoids negative values.

## Game Description

- Players take turns making moves to add or subtract numbers from a shared score.
- Available numbers for moves are shared between the players and include the integers [1, 2, 3, 4, 6, 12].
- Player 1 and Player 2, represented by AI and Human players, respectively, compete to achieve different win conditions.
- Player 1 (AI) aims to secure victory by ensuring that the final score is less than zero.
- Player 2 (Human) seeks to win by achieving a final score greater than zero.
- Each number can be used only once throughout the game.
- The game ends when one of the players achieves their respective win condition or when there are no more moves available.

## Implementation

This Python code implements the "Dodge The Pile" game using the `easyAI` library. The game is represented by the `DodgeThePileGame` class, which inherits from `TwoPlayerGame`. The game logic is implemented in various methods of this class.

## Running
1. Use JetBrains PyCharm
2. Click file > close project if you have any project opened
3. Click 'Get from VCS' button
4. Paste URL: https://github.com/ElectroluxV2/nai1
5. Click clone
6. Right click `main.py`
7. Click run 'main'

## Example gameplay screenshots

![App Screenshot](../screenshots/p1-0.png)

![App Screenshot](../screenshots/p1-1.png)