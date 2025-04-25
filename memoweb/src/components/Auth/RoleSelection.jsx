// src/components/Auth/RoleSelection.jsx
import { useState } from 'react';
import './RoleSelection.css';

const RoleSelection = ({ onRoleSelect }) => {
  const [hoveredRole, setHoveredRole] = useState(null);

  const roles = [
    { 
      id: 'family', 
      label: 'Family Member',
      icon: '👨‍👩‍👧‍👦' 
    },
    { 
      id: 'doctor', 
      label: 'Doctor',
      icon: '👨‍⚕️' 
    },
    { 
      id: 'patient', 
      label: 'Patient',
      icon: '🧑‍🦳',
    }
  ];

  return (
    <div className="role-selection-container">
      <h2>Who are you?</h2>
      <div className="roles-grid">
        {roles.map((role) => (
          <div
            key={role.id}
            className={`role-card ${hoveredRole === role.id ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredRole(role.id)}
            onMouseLeave={() => setHoveredRole(null)}
            onClick={() => onRoleSelect(role.id)}
          >
            <div className="role-icon">{role.icon || role.label.charAt(0)}</div>
            <span>{role.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RoleSelection;