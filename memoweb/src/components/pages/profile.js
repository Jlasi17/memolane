import React, { useEffect, useState } from 'react'

const Profile = () => {
  const [username, setUsername] = useState('')

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem('token')
      if (!token) return

      try {
        const res = await fetch('http://127.0.0.1:8000/api/user', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })

        if (res.ok) {
          const data = await res.json()
          setUsername(data.username)
        } else {
          console.error('Failed to fetch user')
        }
      } catch (err) {
        console.error('Error fetching user:', err)
      }
    }

    fetchUser()
  }, [])

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold">Welcome, {username}!</h2>
    </div>
  )
}

export default Profile