/**
 * Secure client-side encryption for sensitive data (API keys)
 * Uses Web Crypto API for encryption/decryption
 *
 * Security Note: This provides obfuscation against casual inspection
 * but is NOT a replacement for proper server-side key management.
 * Client-side encryption can be reverse-engineered by determined attackers.
 */

const ALGORITHM = 'AES-GCM';
const KEY_LENGTH = 256;
const IV_LENGTH = 12;

/**
 * Generate a device-specific encryption key
 * Uses browser fingerprinting for key derivation
 */
async function getEncryptionKey(): Promise<CryptoKey> {
  const fingerprint = [
    navigator.userAgent,
    navigator.language,
    new Date().getTimezoneOffset().toString(),
    screen.colorDepth.toString(),
    screen.width + 'x' + screen.height,
  ].join('|');

  const encoder = new TextEncoder();
  const data = encoder.encode(fingerprint);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);

  return crypto.subtle.importKey(
    'raw',
    hashBuffer,
    { name: ALGORITHM },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt a string value
 * @param plaintext - The string to encrypt
 * @returns Base64-encoded encrypted data with IV prepended
 */
export async function encryptValue(plaintext: string): Promise<string> {
  if (!plaintext) return '';

  try {
    const key = await getEncryptionKey();
    const encoder = new TextEncoder();
    const data = encoder.encode(plaintext);

    const iv = crypto.getRandomValues(new Uint8Array(IV_LENGTH));

    const encrypted = await crypto.subtle.encrypt(
      { name: ALGORITHM, iv },
      key,
      data
    );

    const combined = new Uint8Array(iv.length + encrypted.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(encrypted), iv.length);

    return btoa(String.fromCharCode(...combined));
  } catch (error) {
    console.error('Encryption error:', error);
    throw new Error('Failed to encrypt value');
  }
}

/**
 * Decrypt a string value
 * @param ciphertext - Base64-encoded encrypted data
 * @returns Decrypted plaintext string
 */
export async function decryptValue(ciphertext: string): Promise<string> {
  if (!ciphertext) return '';

  try {
    const key = await getEncryptionKey();
    const combined = Uint8Array.from(atob(ciphertext), c => c.charCodeAt(0));

    const iv = combined.slice(0, IV_LENGTH);
    const data = combined.slice(IV_LENGTH);

    const decrypted = await crypto.subtle.decrypt(
      { name: ALGORITHM, iv },
      key,
      data
    );

    const decoder = new TextDecoder();
    return decoder.decode(decrypted);
  } catch (error) {
    console.error('Decryption error:', error);
    return '';
  }
}

/**
 * Secure storage wrapper for encrypted values
 */
export class SecureStorage {
  /**
   * Store an encrypted value in sessionStorage
   */
  static async setItem(key: string, value: string): Promise<void> {
    if (!value) {
      sessionStorage.removeItem(key);
      return;
    }

    const encrypted = await encryptValue(value);
    sessionStorage.setItem(key, encrypted);
  }

  /**
   * Retrieve and decrypt a value from sessionStorage
   */
  static async getItem(key: string): Promise<string | null> {
    const encrypted = sessionStorage.getItem(key);
    if (!encrypted) return null;

    try {
      return await decryptValue(encrypted);
    } catch {
      sessionStorage.removeItem(key);
      return null;
    }
  }

  /**
   * Remove an item from sessionStorage
   */
  static removeItem(key: string): void {
    sessionStorage.removeItem(key);
  }

  /**
   * Check if an encrypted item exists
   */
  static hasItem(key: string): boolean {
    return sessionStorage.getItem(key) !== null;
  }
}

/**
 * Add integrity check to detect tampering
 */
export async function generateIntegrityHash(data: string): Promise<string> {
  const encoder = new TextEncoder();
  const dataBuffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Verify integrity hash
 */
export async function verifyIntegrity(data: string, hash: string): Promise<boolean> {
  const computedHash = await generateIntegrityHash(data);
  return computedHash === hash;
}
