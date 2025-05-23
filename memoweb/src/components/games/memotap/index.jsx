import React, { useState, useEffect, useRef } from 'react';
import styles from './MemoTap.module.css';
import { useNavigate } from 'react-router-dom';

const API_BASE_URL = 'http://localhost:8000' || 'http://localhost:8000';

const COLORS = [
  '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
  '#00FFFF', '#FFA500', '#A52A2A', '#800080', '#008000',
  '#000080', '#808000', '#800000', '#008080', '#000000',
  '#C0C0C0', '#808080', '#FFD700', '#D2691E', '#8B008B',
  '#B22222', '#228B22', '#191970', '#8B4513', '#2E8B57'
];

function MemoTap() {
  const [gameState, setGameState] = useState('ready');
  const [round, setRound] = useState(1);
  const [sequence, setSequence] = useState([]);
  const [playerInput, setPlayerInput] = useState([]);
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [numColors, setNumColors] = useState(4);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const animationFrameRef = useRef();
  const colorButtonsRef = useRef([]);
  const navigate = useNavigate();

  // Determine number of colors based on round
  const getNumColors = (roundNum) => {
    if (roundNum <= 5) return 4;
    if (roundNum <= 10) return 6;
    if (roundNum <= 15) return 8;
    if (roundNum <= 20) return 10;
    if (roundNum <= 23) return 16;
    return 25;
  };

  useEffect(() => {
    fetchHighScores();
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const fetchHighScores = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/high-scores?game_name=memotap`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        console.error('Failed to fetch high scores');
        return;
      }
      
      const data = await response.json();
      
      // Safely handle the response
      if (data?.scores?.length > 0) {
        // Find the highest score for memotap
        const memoTapScores = data.scores.filter(score => score.game_name === 'memotap');
        if (memoTapScores.length > 0) {
          const highestScore = Math.max(...memoTapScores.map(score => score.score));
          setHighScore(highestScore);
        } else {
          setHighScore(0);
        }
      } else {
        setHighScore(0); // Default high score if none exists
      }
    } catch (error) {
      console.error('Error fetching high scores:', error);
      setHighScore(0); // Fallback to 0 on error
    }
  };

  const startGame = async () => {
    setGameState('showing');
    setScore(0);
    setRound(1);
    setPlayerInput([]);
    fetchHighScores();
    await generateSequence(1);
  };

  const showSequenceWithDelay = async (sequence) => {
    setGameState('showing');
    
    // Use requestAnimationFrame for smoother timing
    for (let i = 0; i < sequence.length; i++) {
      setHighlightedIndex(sequence[i]);
      
      await new Promise(resolve => {
        animationFrameRef.current = requestAnimationFrame(() => {
          setTimeout(resolve, 800); // Longer display time for better visibility
        });
      });
      
      setHighlightedIndex(-1);
      
      if (i < sequence.length - 1) {
        await new Promise(resolve => {
          animationFrameRef.current = requestAnimationFrame(() => {
            setTimeout(resolve, 300); // Pause between colors
          });
        });
      }
    }
    
    setGameState('playing');
    setPlayerInput([]);
  };

  const generateSequence = async (roundNum) => {
    try {
      const colorsCount = getNumColors(roundNum);
      setNumColors(colorsCount);
      
      const newSequence = Array.from({ length: roundNum }, () => 
        Math.floor(Math.random() * colorsCount)
      );
      setSequence(newSequence);
      
      await showSequenceWithDelay(newSequence);
    } catch (error) {
      console.error('Error generating sequence:', error);
    }
  };

  const handleColorClick = (colorIndex) => {
    if (gameState !== 'playing') return;

    const newPlayerInput = [...playerInput, colorIndex];
    setPlayerInput(newPlayerInput);

    if (sequence[newPlayerInput.length - 1] !== colorIndex) {
      gameOver();
      return;
    }

    if (newPlayerInput.length === sequence.length) {
      roundComplete();
    }
  };

  const roundComplete = () => {
    const newScore = score + round;
    setScore(newScore);
    
    if (round >= 25) {
      gameOver(true);
      return;
    }

    const newRound = round + 1;
    setRound(newRound);
    setGameState('showing');
    generateSequence(newRound);
  };

  const saveScore = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/save-score`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          score: score,
          rounds_completed: round - 1
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        console.error("Failed to save score:", errorData);
        // You could set state here to show an error message to the user
        throw new Error(errorData.detail || 'Failed to save score');
      }
  
      const data = await response.json();
      if (data.is_high_score) {
        // Show a congratulations message for new high score
        console.log("New high score!");
      }
      return data;
    } catch (error) {
      console.error("Error saving score:", error);
      throw error;
    }
  };


  const gameOver = async (completedAll = false) => {
    console.log("Game over triggered"); // Debug log
    setGameState('over');
    if (score > 0 || completedAll) {
      console.log("Attempting to save score..."); // Debug log
      try {
        await saveScore();
        console.log("Score saved successfully after game over"); // Debug log
      } catch (error) {
        console.error("Failed to save score after game over:", error); // Debug log
      }
    }
  };

  const renderColorButtons = () => {
    const currentColors = COLORS.slice(0, numColors);
    return currentColors.map((color, index) => (
      <button
        key={index}
        ref={el => colorButtonsRef.current[index] = el}
        className={`${styles.colorButton} ${highlightedIndex === index ? styles.glowing : ''}`}
        style={{ backgroundColor: color }}
        onClick={() => handleColorClick(index)}
        disabled={gameState !== 'playing'}
      />
    ));
  };

  return (
    <div className={styles.memotapContainer}>
      <button 
        className={styles.backButton}
        onClick={() => navigate('/patient')}
      >
        ← Back to Dashboard
      </button>
      <div className={styles.memotapGame}>
        <h1 className={styles.gameTitle}>MemoTap</h1>
        
        <div className={styles.gameInfo}>
          <p>Score: {score}</p>
          <p>Round: {round}</p>
          <p>High Score: {highScore}</p>
        </div>

        {gameState === 'ready' && (
          <button className={styles.startButton} onClick={startGame}>
            Start Game
          </button>
        )}

        {gameState === 'showing' && (
          <div className={styles.showingSequence}>
            <h2>Memorize the sequence!</h2>
          </div>
        )}

        {gameState === 'playing' && (
          <div className={styles.playing}>
            <h2>Your turn! Tap the sequence</h2>
            <p>Progress: {playerInput.length}/{sequence.length}</p>
          </div>
        )}

        {gameState === 'over' && (
          <div className={styles.gameOver}>
            <h2>Game Over!</h2>
            <p>You reached round {round} with a score of {score}</p>
            <button className={styles.startButton} onClick={startGame}>
              Play Again
            </button>
          </div>
        )}

        <div 
          className={styles.colorGrid} 
          style={{ 
            gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(numColors))}, 1fr)`
          }}
        >
          {renderColorButtons()}
        </div>
      </div>
    </div>
  );
}

export default MemoTap;