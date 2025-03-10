/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import { ConfigService } from '@nestjs/config';
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { compare } from 'bcryptjs';
import { UsersService } from 'src/users/users.service';
import { User } from '@prisma/client';
import { Response } from 'express';
import { JwtService } from '@nestjs/jwt';
import { TokenPayload } from './token-payload-interface';

@Injectable()
export class AuthService {
  constructor(
    private readonly usersService: UsersService,
    private readonly configService: ConfigService,
    private readonly jwtService: JwtService,
  ) {}

  async login(user: User, response: Response) {
    const accessExpirationMs = this.configService.getOrThrow<string>(
      'JWT_ACCESS_TOKEN_EXPIRATION_MS',
    );

    const refreshExpirationMs = this.configService.getOrThrow<string>(
      'JWT_REFRESH_TOKEN_EXPIRATION_MS',
    );

    if (!accessExpirationMs || !refreshExpirationMs) {
      throw new Error(
        'JWT_ACCESS_TOKEN_EXPIRATION_MS or JWT_REFRESH_TOKEN_EXPIRATION_MS not found',
      );
    }

    const expiresAccessToken = new Date();
    expiresAccessToken.setTime(
      expiresAccessToken.getTime() + Number(accessExpirationMs),
    );

    const tokenPayload: TokenPayload = { userId: user.id };

    const accessToken = this.jwtService.sign(tokenPayload, {
      secret: this.configService.getOrThrow<string>('JWT_ACCESS_TOKEN_SECRET'),
      expiresIn: accessExpirationMs,
    });

    // const refreshToken = this.jwtService.sign(tokenPayload, {
    //   secret: this.configService.getOrThrow<string>('JWT_REFRESH_TOKEN_SECRET'),
    //   expiresIn: refreshExpirationMs,
    // });

    response.cookie('Authentication', accessToken, {
      httpOnly: true,
      expires: expiresAccessToken,
    });
  }

  async verifyUser(email: string, password: string) {
    try {
      const user = await this.usersService.getUserByEmail(email);

      const authenticated = await compare(password, user.password);

      if (!authenticated) {
        throw new UnauthorizedException('Invalid credentials');
      }

      return user;
    } catch (error) {
      console.log(error);
      throw new UnauthorizedException('Credentials are not valid');
    }
  }
}
