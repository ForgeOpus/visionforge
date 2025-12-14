/**
 * Authentication context for Firebase authentication
 */
import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  User as FirebaseUser,
  signInWithPopup,
  signOut as firebaseSignOut,
  onAuthStateChanged,
} from 'firebase/auth';
import { auth, googleProvider, githubProvider } from '../lib/firebase';
import { useModelBuilderStore } from '../lib/store';

// User type from our backend
interface VisionForgeUser {
  firebase_uid: string;
  email: string;
  display_name: string | null;
  avatar_url: string | null;
  auth_provider: 'google' | 'github';
  tier: 'free' | 'pro';
  project_count: number;
  created_at: string | null;
  last_login_at: string | null;
}

interface AuthContextType {
  user: VisionForgeUser | null;
  firebaseUser: FirebaseUser | null;
  loading: boolean;
  isGuest: boolean;
  signInWithGoogle: () => Promise<void>;
  signInWithGithub: () => Promise<void>;
  signOut: () => Promise<void>;
  continueAsGuest: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<VisionForgeUser | null>(null);
  const [firebaseUser, setFirebaseUser] = useState<FirebaseUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [isGuest, setIsGuest] = useState(false);

  // Verify token with backend and get user data
  const verifyToken = async (firebaseUser: FirebaseUser): Promise<VisionForgeUser | null> => {
    try {
      const token = await firebaseUser.getIdToken();

      const response = await fetch(`${API_BASE_URL}/api/auth/verify-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token }),
      });

      if (!response.ok) {
        console.error('Failed to verify token with backend');
        return null;
      }

      const data = await response.json();
      return data.user;
    } catch (error) {
      console.error('Error verifying token:', error);
      return null;
    }
  };

  // Refresh user data from backend
  const refreshUser = async () => {
    if (!firebaseUser) return;

    try {
      const token = await firebaseUser.getIdToken();

      const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
      }
    } catch (error) {
      console.error('Error refreshing user:', error);
    }
  };

  // Listen to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setFirebaseUser(firebaseUser);

      if (firebaseUser && !isGuest) {
        // User is signed in
        const backendUser = await verifyToken(firebaseUser);
        setUser(backendUser);
      } else {
        // User is signed out or guest
        setUser(null);
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, [isGuest]);

  // Session management - update session every 5 minutes when user is active
  useEffect(() => {
    if (!firebaseUser || isGuest) return;

    const updateSession = async () => {
      try {
        const token = await firebaseUser.getIdToken();
        await fetch(`${API_BASE_URL}/api/auth/update-session`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ minutes: 5 }), // Track 5 minutes of activity
        });
      } catch (error) {
        console.error('Error updating session:', error);
      }
    };

    // Update session immediately on mount
    updateSession();

    // Then update every 5 minutes
    const intervalId = setInterval(updateSession, 5 * 60 * 1000);

    return () => clearInterval(intervalId);
  }, [firebaseUser, isGuest]);

  // Sign in with Google
  const signInWithGoogle = async () => {
    try {
      setLoading(true);
      setIsGuest(false);
      const result = await signInWithPopup(auth, googleProvider);
      const backendUser = await verifyToken(result.user);
      setUser(backendUser);
    } catch (error) {
      console.error('Error signing in with Google:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Sign in with GitHub
  const signInWithGithub = async () => {
    try {
      setLoading(true);
      setIsGuest(false);
      const result = await signInWithPopup(auth, githubProvider);
      const backendUser = await verifyToken(result.user);
      setUser(backendUser);
    } catch (error) {
      console.error('Error signing in with GitHub:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Sign out
  const signOut = async () => {
    try {
      // Update last login on backend
      if (firebaseUser) {
        const token = await firebaseUser.getIdToken();
        await fetch(`${API_BASE_URL}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
      }

      await firebaseSignOut(auth);
      setUser(null);
      setIsGuest(false);

      // Clear canvas state
      const { reset } = useModelBuilderStore.getState();
      reset();
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  };

  // Continue as guest
  const continueAsGuest = () => {
    setIsGuest(true);
    setUser(null);
    setFirebaseUser(null);
    setLoading(false);

    // Clear canvas state
    const { reset } = useModelBuilderStore.getState();
    reset();
  };

  const value: AuthContextType = {
    user,
    firebaseUser,
    loading,
    isGuest,
    signInWithGoogle,
    signInWithGithub,
    signOut,
    continueAsGuest,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
